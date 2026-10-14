import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import requests

logger = logging.getLogger("SafetyPipeline.Telegram")


class TelegramNotifier:
    """Non-blocking Telegram alerting service using dedicated worker thread."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        config_path: Optional[str] = "configs/telegram_config.json",
        max_queue_size: int = 30
    ):
        self.bot_token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        self.chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "").strip()

        # Load from config file fallback
        if (not self.bot_token or not self.chat_id) and config_path:
            p = Path(config_path)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        token_cfg = cfg.get("telegram_bot_token", "")
                        chat_cfg = cfg.get("telegram_chat_id", "")
                        if not self.bot_token and token_cfg and not token_cfg.startswith("YOUR_"):
                            self.bot_token = token_cfg.strip()
                        if not self.chat_id and chat_cfg and not chat_cfg.startswith("YOUR_"):
                            self.chat_id = str(chat_cfg).strip()
                except Exception as e:
                    logger.warning(f"Could not read Telegram config file: {e}")

        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TelegramWorkerThread"
        )
        self._worker_thread.start()

        if self.is_configured:
            logger.info(f"Telegram alerting active for Chat ID: {self.chat_id}")
        else:
            logger.warning("Telegram Bot Token/Chat ID not configured. Alerts will be logged locally only.")

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _worker_loop(self):
        base_url = f"https://api.telegram.org/bot{self.bot_token}"
        while not self._stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                self.queue.task_done()
                break

            if not self.is_configured:
                self.queue.task_done()
                continue

            try:
                kind = task.get("kind")
                if kind == "message":
                    requests.post(
                        f"{base_url}/sendMessage",
                        data={"chat_id": self.chat_id, "text": task["text"]},
                        timeout=8
                    )
                elif kind == "photo_bytes":
                    requests.post(
                        f"{base_url}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": task.get("caption", "ALERT")},
                        files={"photo": ("alert.jpg", task["bytes"], "image/jpeg")},
                        timeout=12
                    )
                elif kind == "photo_path":
                    with open(task["path"], "rb") as f:
                        requests.post(
                            f"{base_url}/sendPhoto",
                            data={"chat_id": self.chat_id, "caption": task.get("caption", "ALERT")},
                            files={"photo": f},
                            timeout=12
                        )
            except requests.exceptions.Timeout:
                logger.warning("Telegram request timed out. Dropping alert to preserve FPS.")
            except Exception as e:
                logger.error(f"Telegram dispatch error: {e}")
            finally:
                self.queue.task_done()

    def send_message(self, text: str):
        """Enqueue message asynchronously."""
        try:
            self.queue.put_nowait({"kind": "message", "text": text})
        except queue.Full:
            logger.warning("Telegram queue full. Dropped message.")

    def send_photo(self, photo: Union[np.ndarray, str, Path], caption: str = "ALERT"):
        """Enqueue photo asynchronously with direct in-RAM encoding."""
        try:
            if isinstance(photo, np.ndarray):
                success, enc = cv2.imencode(".jpg", photo)
                if success:
                    self.queue.put_nowait({
                        "kind": "photo_bytes",
                        "bytes": enc.tobytes(),
                        "caption": caption
                    })
                else:
                    logger.error("Failed to encode frame to JPEG bytes.")
            else:
                self.queue.put_nowait({
                    "kind": "photo_path",
                    "path": str(photo),
                    "caption": caption
                })
        except queue.Full:
            logger.warning("Telegram queue full. Dropped photo.")

    def stop(self):
        """Stop background worker thread."""
        self._stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker_thread.join(timeout=2.0)
