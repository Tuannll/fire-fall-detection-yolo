import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn


# =========================================================
# PATH RESOLVER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

def resolve_model_path(path_str):
    """Tìm đường dẫn file model trong thư mục hiện tại hoặc các thư mục liên quan."""
    if not path_str:
        return ""
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    candidates = [
        BASE_DIR.parent / "weights" / path_str,
        BASE_DIR / path_str,
        BASE_DIR.parent / path_str,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(p)


# =========================================================
# TELEGRAM — NON-BLOCKING (Queue + Worker Thread)
# =========================================================

def get_telegram_config(cli_token=None, cli_chat_id=None):
    """Lấy Telegram token và chat_id từ CLI, Env Vars, hoặc telegram_config.json."""
    token = cli_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = cli_chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if (not token or not chat_id):
        for candidate_cfg in [BASE_DIR.parent / "configs" / "telegram_config.json", BASE_DIR / "telegram_config.json"]:
            if candidate_cfg.exists():
                try:
                    with open(candidate_cfg, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        cfg_token = cfg.get("telegram_bot_token", "")
                        cfg_chat = cfg.get("telegram_chat_id", "")
                        if not token and cfg_token and not cfg_token.startswith("YOUR_"):
                            token = cfg_token
                        if not chat_id and cfg_chat and not cfg_chat.startswith("YOUR_"):
                            chat_id = cfg_chat
                    break
                except Exception:
                    pass

    return (token or "").strip(), (chat_id or "").strip()


_tg_queue = queue.Queue(maxsize=30)
_tg_config = {"token": "", "chat_id": ""}
_warned_missing_config = False


def init_telegram(token=None, chat_id=None):
    """Khởi tạo cấu hình Telegram."""
    global _tg_config
    t, c = get_telegram_config(token, chat_id)
    _tg_config["token"] = t
    _tg_config["chat_id"] = c
    if not t or not c:
        print("[Telegram] Chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID. Cảnh báo sẽ không gửi qua mạng.")
    else:
        print(f"[Telegram] Đã kết nối bot token (Chat ID: {c})")


def _telegram_worker():
    """Worker chạy ngầm, xử lý queue gửi Telegram trực tiếp từ RAM không qua file disk."""
    global _warned_missing_config
    while True:
        task = _tg_queue.get()
        if task is None:
            break
        try:
            token = _tg_config.get("token")
            chat_id = _tg_config.get("chat_id")
            if not token or not chat_id:
                if not _warned_missing_config:
                    print("[Telegram] Bỏ qua gửi alert: Chưa cấu hình Bot Token / Chat ID.")
                    _warned_missing_config = True
                continue

            base_url = f"https://api.telegram.org/bot{token}"
            kind = task.get("kind")

            if kind == "message":
                requests.post(
                    f"{base_url}/sendMessage",
                    data={"chat_id": chat_id, "text": task["text"]},
                    timeout=8
                )
            elif kind == "photo_bytes":
                requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id, "caption": task.get("caption", "ALERT")},
                    files={"photo": ("alert.jpg", task["bytes"], "image/jpeg")},
                    timeout=12
                )
            elif kind == "photo_path":
                with open(task["path"], "rb") as f:
                    requests.post(
                        f"{base_url}/sendPhoto",
                        data={"chat_id": chat_id, "caption": task.get("caption", "ALERT")},
                        files={"photo": f},
                        timeout=12
                    )
        except requests.exceptions.Timeout:
            print("[Telegram] Timeout — bỏ qua alert, không ảnh hưởng FPS")
        except Exception as e:
            print(f"[Telegram] Lỗi gửi alert: {e}")
        finally:
            _tg_queue.task_done()


_worker_thread = threading.Thread(
    target=_telegram_worker,
    daemon=True,
    name="TelegramWorker"
)
_worker_thread.start()


def send_telegram_message(text):
    """Đưa tin nhắn text vào queue — KHÔNG block vòng lặp chính."""
    try:
        _tg_queue.put_nowait({"kind": "message", "text": text})
    except queue.Full:
        print("[Telegram] Queue đầy — bỏ qua message")


def send_telegram_photo(photo, caption="ALERT"):
    """Đưa ảnh (numpy frame hoặc file path) vào queue — Encode trực tiếp từ RAM, không block vòng lặp chính."""
    try:
        if isinstance(photo, np.ndarray):
            success, enc = cv2.imencode(".jpg", photo)
            if success:
                _tg_queue.put_nowait({"kind": "photo_bytes", "bytes": enc.tobytes(), "caption": caption})
            else:
                print("[Telegram] Lỗi encode ảnh frame")
        else:
            _tg_queue.put_nowait({"kind": "photo_path", "path": str(photo), "caption": caption})
    except queue.Full:
        print("[Telegram] Queue đầy — bỏ qua photo")


# =========================================================
# FALL MODEL
# =========================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (lstm_out * weights.unsqueeze(-1)).sum(dim=1)


class FallLSTM(nn.Module):
    def __init__(self, input_size=102, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = True,
        )

        self.attention = TemporalAttention(hidden * 2)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context     = self.attention(lstm_out)
        return self.head(context)


# =========================================================
# PIPELINE
# =========================================================

class SafetyPipeline:

    def __init__(
        self,
        fire_model_path,
        lstm_model_path,
        device,
        seq_len=30,
        conf_fire=0.45,
        fall_confirm=12,
        fire_interval=1
    ):

        from ultralytics import YOLO

        self.device = device
        self.seq_len = seq_len
        self.conf_fire = conf_fire
        self.fire_interval = max(1, fire_interval)
        self._cached_fire_dets = []

        print("=" * 60)
        print("LOADING MODELS...")
        print("=" * 60)

        # FIRE MODEL
        resolved_fire = resolve_model_path(fire_model_path)
        print(f"Loading fire model: {resolved_fire}...")
        self.fire_model = YOLO(resolved_fire)

        # POSE MODEL
        resolved_pose = resolve_model_path("yolov8n-pose.pt")
        print(f"Loading pose model: {resolved_pose}...")
        self.pose_model = YOLO(resolved_pose)

        # LSTM
        resolved_lstm = resolve_model_path(lstm_model_path)
        print(f"Loading LSTM: {resolved_lstm}...")
        self.lstm = FallLSTM().to(device)

        try:
            state = torch.load(
                resolved_lstm,
                map_location=device,
                weights_only=True
            )
        except TypeError:
            state = torch.load(
                resolved_lstm,
                map_location=device
            )

        self.lstm.load_state_dict(state)
        self.lstm.eval()

        print("ALL MODELS READY!")

        # multi-person tracking buffers and states
        self.track_buffers = {}
        self.track_fall_counters = {}
        self.track_cooldowns = {}
        self.track_probs = {}
        self.track_bboxes = {}
        self.track_last_seen = {}
        self.kp_buffer = deque(maxlen=seq_len)
        self.fall_counter = 0
        self.FALL_CONFIRM = fall_confirm

        self.cooldown_timer = 0
        self.COOLDOWN = 30
        self.max_fall_prob = 0.0

        # stats
        self.frame_count = 0

        # anti spam telegram
        self.last_fire_alert = 0
        self.last_fall_alert = 0

        self.ALERT_INTERVAL = 10


    # =====================================================
    # MULTI-PERSON POSE & FALL TRACKING (ByteTrack)
    # =====================================================

    def process_pose_and_falls(self, frame):
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
            conf=0.3,
            device=self.device,
            verbose=False
        )

        r = results[0]
        annotated = r.plot()
        confirmed_ids = []
        max_prob = 0.0

        if r.boxes is None or len(r.boxes) == 0:
            self._cleanup_stale_tracks()
            return annotated, confirmed_ids, max_prob

        has_ids = (r.boxes.id is not None)
        track_ids = r.boxes.id.int().cpu().tolist() if has_ids else list(range(len(r.boxes)))
        boxes_xyxy = r.boxes.xyxy.int().cpu().numpy()

        if r.keypoints is None or len(r.keypoints) == 0:
            self._cleanup_stale_tracks()
            return annotated, confirmed_ids, max_prob

        xyn_data = r.keypoints.xyn.cpu().numpy()
        conf_data = r.keypoints.conf.cpu().numpy()

        for idx, t_id in enumerate(track_ids):
            if idx >= len(xyn_data) or idx >= len(boxes_xyxy):
                continue

            xy = xyn_data[idx]
            c = conf_data[idx]
            kp = np.concatenate([xy, c[:, None]], axis=1).flatten().astype(np.float32)

            if t_id not in self.track_buffers:
                self.track_buffers[t_id] = deque(maxlen=self.seq_len)
                self.track_fall_counters[t_id] = 0
                self.track_cooldowns[t_id] = 0
                self.track_probs[t_id] = 0.0

            self.track_buffers[t_id].append(kp)
            self.track_bboxes[t_id] = tuple(boxes_xyxy[idx].tolist())
            self.track_last_seen[t_id] = self.frame_count

            if len(self.track_buffers[t_id]) >= self.seq_len:
                kps_51 = np.stack(list(self.track_buffers[t_id]))
                vel = np.zeros_like(kps_51)
                vel[1:] = kps_51[1:] - kps_51[:-1]
                seq_102 = np.concatenate([kps_51, vel], axis=1)

                tensor = torch.tensor(seq_102, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.lstm(tensor)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()

                self.track_probs[t_id] = prob
                max_prob = max(max_prob, prob)

                if self.track_cooldowns[t_id] > 0:
                    self.track_cooldowns[t_id] -= 1
                else:
                    if prob > 0.5:
                        self.track_fall_counters[t_id] += 1
                    else:
                        self.track_fall_counters[t_id] = max(0, self.track_fall_counters[t_id] - 1)

                    if self.track_fall_counters[t_id] >= self.FALL_CONFIRM:
                        self.track_fall_counters[t_id] = 0
                        self.track_cooldowns[t_id] = self.COOLDOWN
                        confirmed_ids.append(t_id)

        self.max_fall_prob = max_prob
        self._cleanup_stale_tracks()
        return annotated, confirmed_ids, max_prob

    def _cleanup_stale_tracks(self):
        stale = [tid for tid, f in self.track_last_seen.items() if (self.frame_count - f) > 150]
        for tid in stale:
            self.track_buffers.pop(tid, None)
            self.track_fall_counters.pop(tid, None)
            self.track_cooldowns.pop(tid, None)
            self.track_probs.pop(tid, None)
            self.track_bboxes.pop(tid, None)
            self.track_last_seen.pop(tid, None)

    # Legacy helper for single-person compatibility
    def get_keypoints(self, frame):
        results = self.pose_model(frame, imgsz=640, conf=0.3, device=self.device, verbose=False)
        r = results[0]
        if r.keypoints is not None and len(r.keypoints.xyn) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            xy = r.keypoints.xyn[best].cpu().numpy()
            conf = r.keypoints.conf[best].cpu().numpy()
            kp = np.concatenate([xy, conf[:, None]], axis=1)
            annotated = r.plot()
        else:
            kp = np.zeros((17, 3), dtype=np.float32)
            annotated = frame.copy()
        return kp.flatten().astype(np.float32), annotated

    def detect_fall(self):
        if len(self.kp_buffer) < self.seq_len:
            return False, 0.0
        kps_51 = np.stack(list(self.kp_buffer))
        vel = np.zeros_like(kps_51)
        vel[1:] = kps_51[1:] - kps_51[:-1]
        seq_102 = np.concatenate([kps_51, vel], axis=1)
        tensor = torch.tensor(seq_102, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.lstm(tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        return prob > 0.5, prob

    def update_fall_state(self, is_fall_raw):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return False
        if is_fall_raw:
            self.fall_counter += 1
        else:
            self.fall_counter = max(0, self.fall_counter - 1)
        if self.fall_counter >= self.FALL_CONFIRM:
            self.fall_counter = 0
            self.cooldown_timer = self.COOLDOWN
            return True
        return False


    # =====================================================
    # FIRE DETECT
    # =====================================================

    def detect_fire(self, frame):

        results = self.fire_model(
            frame,
            imgsz=640,
            conf=self.conf_fire,
            device=self.device,
            verbose=False
        )

        r = results[0]

        detections = []

        if r.boxes is not None:

            for box in r.boxes:

                cls_id = int(box.cls[0])

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                label = r.names[cls_id]

                detections.append(
                    (
                        label,
                        conf,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

        return detections


    # =====================================================
    # OVERLAY
    # =====================================================

    def draw_overlay(
        self,
        frame,
        fire_dets,
        fall_confirmed,
        fall_prob,
        fps
    ):

        h, w = frame.shape[:2]

        overlay = frame.copy()

        # TOP BAR
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 40),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            overlay,
            f"FPS: {fps:.1f} | TRACKS: {len(self.track_buffers)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # MULTI-PERSON TRACK STATUS
        for t_id, bbox in self.track_bboxes.items():
            x1, y1, x2, y2 = bbox
            prob = self.track_probs.get(t_id, 0.0)
            is_falling = (self.track_cooldowns.get(t_id, 0) > 0 or
                          self.track_fall_counters.get(t_id, 0) >= self.FALL_CONFIRM)
            box_color = (0, 0, 255) if is_falling else ((0, 255, 255) if prob > 0.5 else (0, 255, 0))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
            tag = f"ID {t_id}: {'FALL!' if is_falling else f'P:{prob:.2f}'}"
            cv2.putText(overlay, tag, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # FIRE
        for label, conf, x1, y1, x2, y2 in fire_dets:

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                overlay,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ALERTS
        if fire_dets:

            cv2.rectangle(
                overlay,
                (0, 50),
                (w, 110),
                (0, 0, 180),
                -1
            )

            cv2.putText(
                overlay,
                "FIRE DETECTED!",
                (w // 2 - 150, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (255, 255, 255),
                3
            )

        if fall_confirmed:

            cv2.rectangle(
                overlay,
                (0, 120),
                (w, 180),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                overlay,
                "FALL DETECTED!",
                (w // 2 - 150, 160),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (255, 255, 255),
                3
            )

        # FALL BAR (Max probability across all tracked individuals)
        bar_w = int(w * fall_prob)

        cv2.rectangle(
            overlay,
            (0, h - 20),
            (bar_w, h),
            (0, 0, 255) if fall_prob > 0.5 else (0, 255, 0),
            -1
        )

        cv2.putText(
            overlay,
            f"Max Fall Prob: {fall_prob:.2f}",
            (10, h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0
        )

        return frame


    # =====================================================
    # PROCESS
    # =====================================================

    def process(self, frame):

        self.frame_count += 1

        # FIRE (chạy định kỳ theo fire_interval để tối ưu FPS)
        if self.frame_count % self.fire_interval == 0:
            self._cached_fire_dets = self.detect_fire(frame)
        fire_dets = self._cached_fire_dets

        # POSE & MULTI-PERSON TRACKING
        annotated, confirmed_ids, max_prob = self.process_pose_and_falls(frame)
        fall_confirmed = len(confirmed_ids) > 0

        return (
            annotated,
            fire_dets,
            fall_confirmed,
            max_prob
        )


# =========================================================
# MAIN
# =========================================================

def main(args):

    # TELEGRAM INIT
    init_telegram(args.telegram_token, args.telegram_chat_id)

    # DEVICE
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"DEVICE: {device}")

    # RESOLVE & CHECK MODELS
    fire_model = resolve_model_path(args.fire_model)
    lstm_model = resolve_model_path(args.lstm_model)
    for name, p in [("Fire model", fire_model), ("LSTM model", lstm_model)]:
        if not Path(p).exists():
            print(f"MODEL NOT FOUND: {name} -> {p}")
            sys.exit(1)

    # SOURCE
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        print("Cannot open source")

        sys.exit(1)

    # PIPELINE
    pipeline = SafetyPipeline(
        fire_model_path=fire_model,
        lstm_model_path=lstm_model,
        device=device,
        seq_len=args.seq_len,
        conf_fire=args.conf_fire,
        fall_confirm=args.fall_confirm,
        fire_interval=args.fire_interval
    )

    # FPS
    t_prev = time.time()

    fps = 0.0

    print("RUNNING...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # FPS
        t_now = time.time()

        fps = (
            0.9 * fps +
            0.1 * (
                1.0 / max(t_now - t_prev, 1e-6)
            )
        )

        t_prev = t_now

        # PROCESS
        (
            annotated,
            fire_dets,
            fall_confirmed,
            fall_prob
        ) = pipeline.process(frame)

        # OVERLAY
        annotated = pipeline.draw_overlay(
            annotated,
            fire_dets,
            fall_confirmed,
            fall_prob,
            fps
        )

        # =================================================
        # FIRE ALERT
        # =================================================

        if fire_dets:

            if (
                time.time() -
                pipeline.last_fire_alert
            ) > pipeline.ALERT_INTERVAL:

                print("FIRE DETECTED!")

                if args.save_alerts:
                    cv2.imwrite("fire_alert.jpg", annotated)

                send_telegram_photo(
                    annotated,
                    caption="🔥 FIRE DETECTED!"
                )

                pipeline.last_fire_alert = time.time()

        # =================================================
        # FALL ALERT
        # =================================================

        if fall_confirmed:

            if (
                time.time() -
                pipeline.last_fall_alert
            ) > pipeline.ALERT_INTERVAL:

                print("FALL DETECTED!")

                if args.save_alerts:
                    cv2.imwrite("fall_alert.jpg", annotated)

                send_telegram_photo(
                    annotated,
                    caption="🚨 FALL DETECTED!"
                )

                pipeline.last_fall_alert = time.time()

        # SHOW
        cv2.imshow(
            "AI Safety Monitor",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        # QUIT
        if key == ord("q"):
            break

        # SCREENSHOT
        elif key == ord("s"):

            ts = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            fn = f"screenshot_{ts}.jpg"

            cv2.imwrite(fn, annotated)

            print(f"Saved: {fn}")

    # CLEANUP
    cap.release()

    cv2.destroyAllWindows()


# =========================================================
# ARGS
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index hoặc file video"
    )

    parser.add_argument(
        "--fire_model",
        default="fire_best.pt",
        help="File weights YOLO fire"
    )

    parser.add_argument(
        "--lstm_model",
        default="fall_lstm_best.pt",
        help="File weights Fall LSTM"
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"]
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Độ dài sequence frames LSTM"
    )

    parser.add_argument(
        "--conf_fire",
        type=float,
        default=0.45,
        help="Ngưỡng confidence nhận diện lửa"
    )

    parser.add_argument(
        "--fall_confirm",
        type=int,
        default=12,
        help="Số frame liên tiếp xác nhận té ngã"
    )

    parser.add_argument(
        "--fire_interval",
        type=int,
        default=1,
        help="Chu kỳ frame quét lửa (1 = mỗi frame, 2 = cách 1 frame)"
    )

    parser.add_argument(
        "--telegram_token",
        default="",
        help="Telegram Bot Token"
    )

    parser.add_argument(
        "--telegram_chat_id",
        default="",
        help="Telegram Chat ID"
    )

    parser.add_argument(
        "--save_alerts",
        action="store_true",
        help="Lưu ảnh alert ra disk khi phát hiện"
    )

    args = parser.parse_args()

    main(args)
