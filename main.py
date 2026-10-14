import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import time

import cv2
try:
    import yaml
except ImportError:
    yaml = None

from src.alerts.telegram_notifier import TelegramNotifier
from src.pipeline.safety_pipeline import SafetyPipeline


def setup_logging(log_level: str = "INFO", log_file: str = "logs/safety_monitor.log"):
    """Configure structured logging to console and rotating log file."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (10MB max, 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    p = Path(config_path)
    if not p.exists() or yaml is None:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to parse {config_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="AI Real-Time Safety Monitor: Fire & Multi-Person Fall Detection")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--source", default=None, help="Camera index (e.g. 0) or path to video file")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--fire_model", default=None, help="Weights path for YOLO fire model")
    parser.add_argument("--pose_model", default=None, help="Weights path for YOLOv8 pose model")
    parser.add_argument("--lstm_model", default=None, help="Weights path for Fall LSTM model")
    parser.add_argument("--fire_interval", type=int, default=None, help="Fire inference period (frames)")
    parser.add_argument("--fall_confirm", type=int, default=None, help="Consecutive frames to confirm fall")
    parser.add_argument("--telegram_token", default=None, help="Telegram bot token")
    parser.add_argument("--telegram_chat_id", default=None, help="Telegram chat ID")
    parser.add_argument("--save_alerts", action="store_true", help="Save alert snapshot images to disk")
    parser.add_argument("--no_gui", action="store_true", help="Run in headless server mode without cv2.imshow")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = cfg.get("pipeline", {})
    model_cfg = cfg.get("models", {})
    tg_cfg = cfg.get("telegram", {})
    log_cfg = cfg.get("logging", {})

    setup_logging(
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/safety_monitor.log")
    )
    logger = logging.getLogger("Main")
    logger.info("Starting AI Safety Monitor v2.0...")

    # Resolve parameters
    source = args.source if args.source is not None else pipe_cfg.get("source", "0")
    source_val = int(source) if str(source).isdigit() else str(source)
    device = args.device or pipe_cfg.get("device", "auto")

    fire_model = args.fire_model or model_cfg.get("fire_model", "weights/fire_best.pt")
    pose_model = args.pose_model or model_cfg.get("pose_model", "weights/yolov8n-pose.pt")
    lstm_model = args.lstm_model or model_cfg.get("lstm_model", "weights/fall_lstm_best.pt")

    fire_interval = args.fire_interval if args.fire_interval is not None else pipe_cfg.get("fire_interval", 2)
    fall_confirm = args.fall_confirm if args.fall_confirm is not None else pipe_cfg.get("fall_confirm", 12)
    seq_len = pipe_cfg.get("seq_len", 30)
    fire_conf = pipe_cfg.get("fire_conf", 0.45)
    fall_threshold = pipe_cfg.get("fall_threshold", 0.5)

    notifier = TelegramNotifier(
        bot_token=args.telegram_token or tg_cfg.get("bot_token"),
        chat_id=args.telegram_chat_id or tg_cfg.get("chat_id"),
        config_path=tg_cfg.get("config_file", "configs/telegram_config.json")
    )

    pipeline = SafetyPipeline(
        fire_model_path=fire_model,
        lstm_model_path=lstm_model,
        pose_model_path=pose_model,
        device=device,
        seq_len=seq_len,
        fall_confirm=fall_confirm,
        fall_threshold=fall_threshold,
        fire_conf=fire_conf,
        fire_interval=fire_interval,
        notifier=notifier,
        save_alerts=args.save_alerts
    )

    logger.info(f"Connecting to video source: {source_val}")
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source_val}")
        sys.exit(1)

    fps = 0.0
    t_prev = time.time()
    logger.info("Inference loop running. Press 'q' to quit, 's' for screenshot.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video stream ended or frame read failed.")
                break

            t_now = time.time()
            dt = max(t_now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t_prev = t_now

            annotated_frame, fire_dets, confirmed_fall_ids = pipeline.process_frame(frame)
            display_frame = pipeline.draw_dashboard_overlay(
                annotated_frame,
                fire_dets,
                confirmed_fall_ids,
                fps
            )

            if not args.no_gui:
                cv2.imshow("AI Safety Monitor v2.0 - Fire & Multi-Person Fall", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed by user.")
                    break
                elif key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    snap_path = f"logs/screenshot_{ts}.jpg"
                    cv2.imwrite(snap_path, display_frame)
                    logger.info(f"Saved screenshot: {snap_path}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user (SIGINT).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        notifier.stop()
        logger.info("Pipeline shut down successfully.")


if __name__ == "__main__":
    main()
