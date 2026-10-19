<div align="center">

# 🔥 AI Safety Monitor

**Real-time safety surveillance system: Multi-person fall detection and fire/smoke recognition using YOLOv8, ByteTrack, and Bi-LSTM with Temporal Attention.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [Tiếng Việt](README.vi.md)

**[✨ Features](#features) • [🏛️ Architecture](#architecture) • [🚀 Quick Start](#quick-start) • [💻 Usage](#usage) • [📊 Performance](#performance) • [❓ FAQ](#faq)**

</div>

---

<a id="features"></a>
## ✨ Features

| Domain | Capability | Technical Details |
| :--- | :--- | :--- |
| **Fire & Smoke Detection** | Continuous flame and smoke monitoring | Custom YOLOv8s with intermittent interval inference optimizing inference FPS |
| **Multi-Person Tracking** | Unique individual identification | ByteTrack assigns persistent `track_id` values avoiding ID switches |
| **Fall Detection** | Posture transition & impact classification | 2-layer Bi-LSTM + Temporal Attention over 102D feature vectors ($17 \times 3$ keypoints + velocity $\Delta v$) |
| **Instant Alerts** | Real-time Telegram Bot notifications | Non-blocking thread-safe queue with in-memory JPEG compression avoiding frame drops |
| **Production Ready** | Local or server deployment | Centralized YAML configs, Docker, Docker Compose, rotating logs |

---

<a id="architecture"></a>
## 🏛️ System Architecture

```text
[ Camera / RTSP / Video Stream ]
               │
               ▼
   [ Frame Preprocessing (640x640) ]
       │                        │
       ▼ (Every N frames)       ▼ (Every frame)
[ YOLO Fire & Smoke ]      [ YOLOv8-Pose + ByteTrack ]
       │                        │
       │ (Bounding Boxes)       ├── Track #1: Bbox + 17 KPs (x, y, conf)
       │                        ├── Track #2: Bbox + 17 KPs (x, y, conf)
       │                        └── Track #k: Bbox + 17 KPs (x, y, conf)
       │                                     │
       │                                     ▼
       │                        [ Multi-Person State Buffer ]
       │                        • Deque(maxlen=30) per Track ID
       │                        • 102D Feature vector: [KPs (51) | ΔKPs (51)]
       │                                     │
       │                                     ▼
       │                        [ Bi-LSTM + Temporal Attention ]
       │                        • Per-person fall probability
       │                        • Persistence confirmation counter (12 frames)
       │                                     │
       └───────────────────┬─────────────────┘
                           │
                           ▼
          [ High-Contrast Dashboard Overlay ]
          (Track BBoxes, Status Labels, FPS, Alert Banners)
                           │
                           ▼
         [ Non-Blocking RAM Alert Dispatcher ]
       (Thread-safe Queue -> In-Memory JPEG -> Telegram Bot)
```

---

<a id="quick-start"></a>
## 🚀 Quick Start

### 1. Environment Setup

Requirement: **Python 3.10** (Conda recommended).

```bash
# Create and activate environment
conda create -n cv_safety python=3.10 -y
conda activate cv_safety

# Install dependencies
python -m pip install -r requirements.txt
```

> [!TIP]
> For NVIDIA GPU systems, install CUDA-accelerated PyTorch for optimal FPS:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```

### 2. Model Weights Setup

Place the 3 required weights in the `weights/` directory:
- `weights/fire_best.pt` (~22.5 MB)
- `weights/fall_lstm_best.pt` (~2.6 MB)
- `weights/yolov8n-pose.pt` (~6.8 MB)

See details in [`weights/README.md`](weights/README.md).

### 3. Telegram Notifications Setup (Optional)

Copy and configure environment variables:

```bash
cp .env.example .env
# Edit .env:
# TELEGRAM_BOT_TOKEN=your_token
# TELEGRAM_CHAT_ID=your_chat_id
```

Alternatively, populate `configs/telegram_config.json` following [`configs/telegram_config.example.json`](configs/telegram_config.example.json).

---

<a id="usage"></a>
## 💻 Usage

### Run with Webcam
```bash
# Auto hardware selection (CUDA preferred):
python main.py --source 0 --device auto

# Enforce NVIDIA GPU:
python main.py --source 0 --device cuda
```

### Run with Video File
```bash
python main.py --source "path/to/test_video.mp4" --device auto
```

### Headless Server Mode (RTSP Stream)
```bash
python main.py --source "rtsp://camera_ip_address" --no_gui
```

### Keyboard Shortcuts
- `q`: Exit application safely.
- `s`: Save dashboard screenshot to `logs/`.

---

<a id="docker"></a>
## 🐳 Docker Deployment

```bash
# Launch background container
docker compose up -d

# Follow real-time logs
docker compose logs -f

# Stop container
docker compose down
```

---

<a id="testing"></a>
## 🧪 Automated Testing

Validate model architecture, multi-person tracking algorithms, and non-blocking alerting:

```bash
python -m unittest tests/test_safety_system.py
```

---

<a id="performance"></a>
## 📊 Performance Benchmarks

Measured on $640 \times 640$ video streams:

| Hardware | Configuration | FPS | Notes |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU (RTX / GTX / CUDA)** | `device=cuda`, `fire_interval=2` | **~30 - 50 FPS** | Smooth real-time tracking for multiple people |
| **Intel / AMD CPU** | `device=cpu`, `fire_interval=5`, `imgsz=416` | **~18 - 25 FPS** | Optimized with relaxed fire detection intervals |
| **Intel / AMD CPU** | `device=cpu`, `fire_interval=2`, `imgsz=640` | **~8 - 10 FPS** | Default full-resolution settings |

---

<a id="faq"></a>
## ❓ FAQ

<details>
<summary><b>1. Does sending Telegram alerts cause camera stutter?</b></summary>
<br/>

No. Snapshots are encoded to JPEG in RAM and enqueued into a thread-safe `queue.Queue`. A dedicated worker thread manages HTTP transmission independently.
</details>

<details>
<summary><b>2. How does the system avoid false alarms when picking up dropped objects?</b></summary>
<br/>

Temporal Attention combined with a 12-frame confirmation threshold distinguishes intentional bending (gradual deceleration) from uncontrolled free fall (sudden velocity spike).
</details>

<details>
<summary><b>3. How are multiple people tracked simultaneously?</b></summary>
<br/>

ByteTrack assigns distinct `track_id` tags. Each person maintains an independent 30-frame keypoint buffer and verification counter.
</details>

---

<a id="license"></a>
## 📄 License

Distributed under the [MIT License](LICENSE).
