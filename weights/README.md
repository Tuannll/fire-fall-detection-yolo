# 🧠 Model Checkpoints & Pre-trained Weights

Thư mục này dùng để lưu trữ các trọng số mô hình đã huấn luyện. Do kích thước lớn của các file nhị phân (`.pt`, `.onnx`), chúng được cấu hình trong `.gitignore` để không đẩy lên Git repository công khai.

---

## 📦 Danh Sách Model Cần Thiết

| File | Kích thước | Mô tả | Nguồn gốc |
| :--- | :--- | :--- | :--- |
| **`fire_best.pt`** | ~22.5 MB | YOLOv8s fine-tuned phát hiện Lửa & Khói (2 classes: `fire`, `smoke`) | Huấn luyện từ `notebooks/01_fire_smoke_detection_yolo.ipynb` |
| **`fall_lstm_best.pt`** | ~2.6 MB | Bi-LSTM 2 tầng + Temporal Attention nhận diện té ngã từ vector đặc trưng 102D | Huấn luyện từ `notebooks/03_train_bilstm_attention_fall.ipynb` |
| **`yolov8n-pose.pt`** | ~6.8 MB | YOLOv8n Pose Estimation trích xuất 17 keypoints cơ thể người | Ultralytics Official Release (tự động tải nếu thiếu) |

---

## 📥 Hướng Dẫn Tải & Cài Đặt Weights

### Cách 1: Tải tự động qua script (Khuyến nghị)
Chạy script tự động quét và gom weights:
```bash
python scripts/setup_weights.py
```

### Cách 2: Tải thủ công từ đám mây / GitHub Releases
1. Tải bộ trọng số mô hình từ kho lưu trữ:
   - **GitHub Releases:** [https://github.com/Tuannll/fire-fall-detection-yolo/releases](https://github.com/Tuannll/fire-fall-detection-yolo/releases)
2. Tải và đặt trực tiếp 3 file `.pt` vào thư mục này:
   ```
   weights/
   ├── .gitkeep
   ├── fire_best.pt
   ├── fall_lstm_best.pt
   └── yolov8n-pose.pt
   ```

### Cách 3: Huấn luyện lại từ đầu (Scratch Training)
- Chạy lần lượt các notebook trong thư mục `notebooks/` trên Kaggle GPU (T4 x 2) hoặc máy trạm có CUDA:
  1. `01_fire_smoke_detection_yolo.ipynb` → sinh ra `fire_best.pt`
  2. `02_extract_pose_keypoints.ipynb` → trích xuất dataset vector 102 chiều `keypoints.zip`
  3. `03_train_bilstm_attention_fall.ipynb` → sinh ra `fall_lstm_best.pt`
