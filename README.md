<div align="center">

# 🔥 AI Safety Monitor

**Hệ thống giám sát an toàn thời gian thực: Phát hiện cháy nổ & té ngã đa đối tượng sử dụng YOLOv8, ByteTrack và Bi-LSTM Attention.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**[Tính năng](#-tính-năng) • [Kiến trúc](#-kiến-trúc-hệ-thống) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-sử-dụng) • [Hiệu năng](#-hiệu-năng) • [FAQ](#-faq)**

</div>

---

## ✨ Tính năng

| Nhóm | Chức năng | Chi tiết kỹ thuật |
| :--- | :--- | :--- |
| **Phát hiện cháy & khói** | Quét lửa và cột khói liên tục | YOLOv8s custom, intermittent inference tối ưu FPS |
| **Tracking đa người** | Định danh từng người trong khung hình | ByteTrack gán `track_id` độc lập, không nhảy đối tượng |
| **Nhận diện té ngã** | Phân loại tư thế rơi và tiếp đất | Bi-LSTM 2 tầng + Temporal Attention trên vector 102D ($17 \times 3$ keypoints + vận tốc $\Delta v$) |
| **Cảnh báo tức thời** | Báo động qua Telegram Bot | Hàng đợi non-blocking, nén ảnh JPEG trên RAM, không nghẽn luồng camera |
| **Đóng gói chuẩn** | Chạy local hoặc server | Cấu hình YAML tập trung, Docker, Docker Compose, logging xoay vòng |

---

## 📌 Kiến trúc hệ thống

```
[ Camera / RTSP / Video Stream ]
               │
               ▼
   [ Frame Preprocessing (640x640) ]
       │                        │
       ▼ (Every N frames)       ▼ (Mỗi frame)
[ YOLO Fire & Smoke ]      [ YOLOv8-Pose + ByteTrack ]
       │                        │
       │ (Bounding Boxes)       ├── Track #1: Bbox + 17 KPs (x, y, conf)
       │                        ├── Track #2: Bbox + 17 KPs (x, y, conf)
       │                        └── Track #k: Bbox + 17 KPs (x, y, conf)
       │                                     │
       │                                     ▼
       │                        [ Multi-Person State Buffer ]
       │                        • Deque(maxlen=30) từng Track ID
       │                        • Vector đặc trưng 102D: [KPs (51) | ΔKPs (51)]
       │                                     │
       │                                     ▼
       │                        [ Bi-LSTM + Temporal Attention ]
       │                        • Xác suất té ngã từng người
       │                        • Bộ đếm xác nhận (12 frames liên tiếp)
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

## 🚀 Cài đặt

### 1. Chuẩn bị môi trường

Yêu cầu: **Python 3.10** (khuyến nghị dùng Conda).

```bash
# Tạo và kích hoạt môi trường
conda create -n cv_safety python=3.10 -y
conda activate cv_safety

# Cài đặt thư viện dependencies
python -m pip install -r requirements.txt
```

> [!TIP]
> Nếu máy có card rời NVIDIA, cài đặt bản PyTorch CUDA để đạt FPS tối đa:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
> ```

### 2. Thiết lập trọng số mô hình (Weights)

Tải hoặc đặt 3 file trọng số vào thư mục `weights/`:
- `weights/fire_best.pt` (~22.5 MB)
- `weights/fall_lstm_best.pt` (~2.6 MB)
- `weights/yolov8n-pose.pt` (~6.8 MB)

Chi tiết xem tại [`weights/README.md`](weights/README.md).

### 3. Cấu hình Telegram Alert (Tùy chọn)

Sao chép file mẫu và cấu hình:

```bash
cp .env.example .env
# Chỉnh sửa .env:
# TELEGRAM_BOT_TOKEN=your_token
# TELEGRAM_CHAT_ID=your_chat_id
```

Hoặc điền vào file `configs/telegram_config.json` theo mẫu [`configs/telegram_config.example.json`](configs/telegram_config.example.json).

---

## 💻 Sử dụng

### Chạy trực tiếp với Webcam
```bash
# Tự động chọn thiết bị (ưu tiên GPU CUDA):
python main.py --source 0 --device auto

# Chạy ép buộc trên GPU NVIDIA:
python main.py --source 0 --device cuda
```

### Chạy với Video kiểm thử
```bash
python main.py --source "path/to/test_video.mp4" --device auto
```

### Chạy Server không giao diện (Headless Mode)
Dành cho máy chủ đám mây, luồng camera IP (RTSP) hoặc container Docker:
```bash
python main.py --source "rtsp://camera_ip_address" --no_gui
```

### Phím tắt điều khiển
- `q`: Thoát chương trình an toàn.
- `s`: Chụp ảnh màn hình dashboard và lưu vào thư mục `logs/`.

---

## 🐳 Triển khai Docker

```bash
# Khởi chạy dịch vụ nền
docker compose up -d

# Xem log theo dõi thời gian thực
docker compose logs -f

# Dừng dịch vụ
docker compose down
```

---

## 🧪 Kiểm thử tự động (Unit Tests)

Bộ test tự động xác thực kiến trúc model, thuật toán tracking đa người và hàng đợi bất đồng bộ:

```bash
python -m unittest tests/test_safety_system.py
```

```
Ran 7 tests in 0.280s
OK
```

---

## 📁 Cấu trúc thư mục

```
pj-tgmt/
├── configs/
│   ├── config.yaml                     # Cấu hình ngưỡng, chu kỳ, đường dẫn
│   └── telegram_config.example.json    # Mẫu cấu hình bot Telegram
├── docs/                               # Tài liệu học thuật & báo cáo đồ án
│   ├── report.pdf                      # Báo cáo đồ án chi tiết
│   ├── presentation_slides.pdf         # Slide báo cáo thuyết minh
│   └── training_metrics/               # Biểu đồ PR, F1, Loss, Confusion Matrix
├── legacy/
│   └── test_single_file.py             # Script monolithic (tương thích ngược)
├── notebooks/                          # Jupyter Notebooks huấn luyện Kaggle T4x2
│   ├── 01_fire_smoke_detection_yolo.ipynb
│   ├── 02_extract_pose_keypoints.ipynb
│   ├── 03_train_bilstm_attention_fall.ipynb
│   └── README.md
├── scripts/
│   ├── setup_weights.py                # Script quét và gom weights
│   ├── auto_label_fire_smoke.py        # Tool gán nhãn tự động bằng YOLO-World
│   └── reorganize_repo.py              # Script dọn dẹp thư mục rác
├── src/                                # Mã nguồn kiến trúc Clean Architecture
│   ├── alerts/
│   │   └── telegram_notifier.py        # Worker thread cảnh báo bất đồng bộ
│   ├── models/
│   │   └── fall_lstm.py                # Bi-LSTM + Temporal Attention
│   ├── pipeline/
│   │   └── safety_pipeline.py          # Pipeline điều phối inference & Dashboard UI
│   └── tracker/
│       └── person_tracker.py           # ByteTrack đa người & vi phân vận tốc
├── tests/
│   └── test_safety_system.py           # Bộ unit test tự động
├── weights/
│   ├── .gitkeep
│   └── README.md                       # Hướng dẫn tải weights
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── main.py                             # CLI entrypoint chính
├── requirements.txt
└── README.md
```

---

## 📊 Hiệu năng

Đo lường trên video độ phân giải $640 \times 640$:

| Phần cứng | Thiết lập | FPS | Ghi chú |
| :--- | :--- | :--- | :--- |
| **GPU NVIDIA (RTX / GTX / CUDA)** | `device=cuda`, `fire_interval=2` | **~30 - 50 FPS** | Mượt thời gian thực, xử lý nhiều người đồng thời |
| **CPU Intel / AMD** | `device=cpu`, `fire_interval=5`, `imgsz=416` | **~18 - 25 FPS** | Đã tối ưu giãn cách inference lửa |
| **CPU Intel / AMD** | `device=cpu`, `fire_interval=2`, `imgsz=640` | **~8 - 10 FPS** | Cấu hình mặc định |

---

## ❓ FAQ

<details>
<summary><b>1. Hệ thống có bị giật/lag khi gửi cảnh báo Telegram không?</b></summary>

Không. Ảnh được nén JPEG trực tiếp trên RAM và đưa vào `queue.Queue`. Một luồng nền độc lập gửi HTTP request, camera loop không bị dừng ngay cả khi mất mạng.
</details>

<details>
<summary><b>2. Tại sao người cúi xuống nhặt đồ không bị báo ngã nhầm?</b></summary>

Cơ chế Temporal Attention kết hợp ngưỡng xác nhận 12 frames liên tiếp giúp phân biệt được động tác chủ động (vận tốc giảm dần) với sự cố rơi tự do mất kiểm soát (vận tốc biến thiên đột ngột).
</details>

<details>
<summary><b>3. Khung hình có 3-4 người thì theo dõi thế nào?</b></summary>

ByteTrack gán `track_id` riêng biệt cho từng người. Mỗi người sở hữu bộ đệm 30 frame keypoint và bộ đếm xác nhận ngã độc lập, không bị lẫn lộn trạng thái.
</details>

---

## 📄 License

Dự án phát hành theo giấy phép [MIT License](LICENSE).
