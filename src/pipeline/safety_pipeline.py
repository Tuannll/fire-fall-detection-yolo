import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.alerts.telegram_notifier import TelegramNotifier
from src.models.fall_lstm import FallLSTM
from src.tracker.person_tracker import MultiPersonTracker

logger = logging.getLogger("SafetyPipeline")


def resolve_model_path(path_str: str, base_dir: Optional[Path] = None) -> str:
    """Resolve model weight path across project directories."""
    if not path_str:
        return ""
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    base = base_dir or Path(__file__).resolve().parents[2]
    candidates = [
        base / path_str,
        base / "weights" / p.name,
        base / "Source" / p.name,
        base / "Source" / "New folder" / p.name,
        base.parent / "Computer-Vision" / "Models" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(p)


class SafetyPipeline:
    """Core real-time safety monitoring pipeline for fire and multi-person fall detection."""

    def __init__(
        self,
        fire_model_path: str,
        lstm_model_path: str,
        pose_model_path: str = "yolov8n-pose.pt",
        device: str = "auto",
        seq_len: int = 30,
        fall_confirm: int = 12,
        fall_threshold: float = 0.5,
        fire_conf: float = 0.45,
        fire_interval: int = 2,
        pose_conf: float = 0.3,
        imgsz: int = 640,
        notifier: Optional[TelegramNotifier] = None,
        alert_interval: float = 10.0,
        save_alerts: bool = False
    ):
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.seq_len = seq_len
        self.fire_conf = fire_conf
        self.fire_interval = max(1, fire_interval)
        self.pose_conf = pose_conf
        self.imgsz = imgsz
        self.alert_interval = alert_interval
        self.save_alerts = save_alerts
        self.notifier = notifier or TelegramNotifier()

        logger.info(f"Initializing SafetyPipeline on device: {self.device}")

        # 1. Fire model
        resolved_fire = resolve_model_path(fire_model_path)
        logger.info(f"Loading Fire model from: {resolved_fire}")
        self.fire_model = YOLO(resolved_fire)

        # 2. Pose model
        resolved_pose = resolve_model_path(pose_model_path)
        logger.info(f"Loading Pose tracker model from: {resolved_pose}")
        self.pose_model = YOLO(resolved_pose)

        # 3. Fall LSTM model
        resolved_lstm = resolve_model_path(lstm_model_path)
        logger.info(f"Loading FallLSTM model from: {resolved_lstm}")
        self.lstm = FallLSTM().to(self.device)

        try:
            state = torch.load(resolved_lstm, map_location=self.device, weights_only=True)
        except TypeError:
            # Fallback for legacy PyTorch versions (< 2.4) that don't support weights_only parameter
            state = torch.load(resolved_lstm, map_location=self.device)

        self.lstm.load_state_dict(state)
        self.lstm.eval()

        # 4. Multi-person tracker
        self.tracker = MultiPersonTracker(
            seq_len=seq_len,
            fall_confirm=fall_confirm,
            fall_threshold=fall_threshold
        )

        # State cache
        self.frame_count = 0
        self._cached_fire_dets: List[Tuple[str, float, int, int, int, int]] = []
        self.last_fire_alert_time = 0.0
        self.last_fall_alert_time: Dict[int, float] = {}

    def detect_fire(self, frame: np.ndarray) -> List[Tuple[str, float, int, int, int, int]]:
        """Run YOLO fire/smoke detection."""
        results = self.fire_model(
            frame,
            imgsz=self.imgsz,
            conf=self.fire_conf,
            device=self.device,
            verbose=False
        )
        r = results[0]
        detections = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[cls_id]
                detections.append((label, conf, x1, y1, x2, y2))
        return detections

    def process_pose_and_falls(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """Track multiple people and evaluate fall risk per track."""
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=self.pose_conf,
            device=self.device,
            verbose=False
        )
        r = results[0]
        annotated_frame = r.plot()
        confirmed_fall_ids: List[int] = []

        if r.boxes is None or len(r.boxes) == 0:
            self.tracker.cleanup_stale_tracks(self.frame_count)
            return annotated_frame, confirmed_fall_ids

        has_ids = (r.boxes.id is not None)
        track_ids = r.boxes.id.int().cpu().tolist() if has_ids else list(range(len(r.boxes)))
        boxes_xyxy = r.boxes.xyxy.int().cpu().numpy()

        if r.keypoints is None or len(r.keypoints) == 0:
            self.tracker.cleanup_stale_tracks(self.frame_count)
            return annotated_frame, confirmed_fall_ids

        xyn_data = r.keypoints.xyn.cpu().numpy()
        conf_data = r.keypoints.conf.cpu().numpy()

        for idx, track_id in enumerate(track_ids):
            if idx >= len(xyn_data) or idx >= len(boxes_xyxy):
                continue

            xy = xyn_data[idx]      # (17, 2)
            c = conf_data[idx]      # (17,)
            kp = np.concatenate([xy, c[:, None]], axis=1).flatten().astype(np.float32)
            bbox = tuple(boxes_xyxy[idx].tolist())

            seq_102 = self.tracker.update_keypoints(track_id, kp, bbox, self.frame_count)
            if seq_102 is not None:
                tensor = torch.from_numpy(seq_102).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.lstm(tensor)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()

                is_confirmed = self.tracker.update_fall_evaluation(track_id, prob)
                if is_confirmed:
                    confirmed_fall_ids.append(track_id)

        self.tracker.cleanup_stale_tracks(self.frame_count)
        return annotated_frame, confirmed_fall_ids

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple], List[int]]:
        """Complete single frame inference pipeline."""
        self.frame_count += 1

        # 1. Fire detection (decimated by fire_interval)
        if self.frame_count % self.fire_interval == 0:
            self._cached_fire_dets = self.detect_fire(frame)
        fire_dets = self._cached_fire_dets

        # 2. Multi-person pose & fall detection
        annotated_frame, confirmed_fall_ids = self.process_pose_and_falls(frame)

        # 3. Handle alerts
        now = time.time()
        if fire_dets and (now - self.last_fire_alert_time) > self.alert_interval:
            logger.warning(f"Fire/Smoke detected! Sending alert. Total detections: {len(fire_dets)}")
            if self.save_alerts:
                cv2.imwrite("logs/fire_alert.jpg", annotated_frame)
            self.notifier.send_message("🔥 [CẢNH BÁO] Phát hiện LỬA / KHÓI tại khu vực camera!")
            self.notifier.send_photo(annotated_frame, caption="🔥 CẢNH BÁO CHÁY / KHÓI!")
            self.last_fire_alert_time = now

        for track_id in confirmed_fall_ids:
            last_alert = self.last_fall_alert_time.get(track_id, 0.0)
            if (now - last_alert) > self.alert_interval:
                logger.warning(f"Human Fall CONFIRMED for Person ID #{track_id}! Dispatching alert.")
                if self.save_alerts:
                    cv2.imwrite(f"logs/fall_alert_person_{track_id}.jpg", annotated_frame)
                msg = f"🚨 [CẢNH BÁO TÉ NGÃ] Phát hiện người bị ngã! Đối tượng ID #{track_id}"
                self.notifier.send_message(msg)
                self.notifier.send_photo(annotated_frame, caption=msg)
                self.last_fall_alert_time[track_id] = now

        return annotated_frame, fire_dets, confirmed_fall_ids

    def draw_dashboard_overlay(
        self,
        frame: np.ndarray,
        fire_dets: List[Tuple],
        confirmed_fall_ids: List[int],
        fps: float
    ) -> np.ndarray:
        """Render high-contrast visual dashboard overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 42), (18, 18, 18), -1)
        active_count = len([t for t in self.tracker.tracks.values() if self.frame_count - t.last_seen_frame <= 2])
        status_text = f"FPS: {fps:4.1f} | Device: {self.device} | People tracked: {active_count} | Frame: {self.frame_count}"
        cv2.putText(overlay, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        # Draw per-person tracking boxes and labels
        for t_id, track in self.tracker.tracks.items():
            if self.frame_count - track.last_seen_frame > 2 or track.bbox is None:
                continue

            x1, y1, x2, y2 = track.bbox
            if track.is_fall_confirmed:
                color = (0, 0, 255)       # Red
                status_label = f"ID #{t_id} FALL CONFIRMED!"
            elif track.fall_prob > 0.5:
                color = (0, 165, 255)     # Orange
                status_label = f"ID #{t_id} WARN ({track.fall_prob:.2f})"
            else:
                color = (0, 255, 0)       # Green
                status_label = f"ID #{t_id} ({track.fall_prob:.2f})"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, status_label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Fire bounding boxes
        for label, conf, x1, y1, x2, y2 in fire_dets:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 240), 2)
            cv2.putText(overlay, f"{label}: {conf:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Banner alerts
        if fire_dets:
            cv2.rectangle(overlay, (0, 50), (w, 105), (0, 0, 180), -1)
            cv2.putText(overlay, "🔥 FIRE DETECTED!", (w // 2 - 160, 88), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3)

        if confirmed_fall_ids:
            banner_y = 115 if fire_dets else 50
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + 55), (0, 0, 220), -1)
            txt = f"🚨 FALL DETECTED! (ID: {', '.join(map(str, confirmed_fall_ids))})"
            cv2.putText(overlay, txt, (w // 2 - 220, banner_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
