# 📓 Training & Data Preparation Notebooks

Thư mục chứa các Jupyter Notebook phục vụ huấn luyện mô hình trên Kaggle GPU (T4 x 2):

1. **`01_fire_smoke_detection_yolo.ipynb`**:
   - Huấn luyện YOLOv8s trên tập dữ liệu Fire & Smoke.
   - Xuất biểu đồ PR curve, Confusion Matrix, và trọng số `fire_best.pt`.
2. **`02_extract_pose_keypoints.ipynb`**:
   - Trích xuất 17 keypoint cơ thể người từ chuỗi video dataset ngã.
   - Trích xuất vận tốc biến thiên $\Delta v$ và lưu thành dataset vector đặc trưng 102 chiều.
3. **`03_train_bilstm_attention_fall.ipynb`**:
   - Định nghĩa mô hình Bi-LSTM + Temporal Attention.
   - Huấn luyện với class weights cân bằng mẫu, lưu checkpoint `fall_lstm_best.pt`.
