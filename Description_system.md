# Hệ Thống Tìm Kiếm Hoa (FlowerSearch System)

## 1. Tổng Quan

Hệ thống tìm kiếm hoa dựa trên hình ảnh sử dụng deep learning và vector similarity search.

## 2. Cấu Trúc Thư Mục

```
FlowerSearch/
├── Dataset/ # Chứa 20 loại hoa (Allium, Anemone, Calla Lily,...)
├── chroma_db/ # Lưu trữ vector database
├── static/
│ └── uploads/ # Lưu ảnh upload
├── app.py # FastAPI backend
├── main.py # FastAPI server
├── streamlit_app.py # Frontend interface
├── image_processor.py # Xử lý ảnh
├── feature_extractor.py # Trích xuất đặc trưng
├── vector_store.py # Quản lý vector database
├── evaluation.py # Đánh giá hệ thống
├── init_db.py # Khởi tạo database
├── requirements.txt # Dependencies
└── README.md # Tài liệu hướng dẫn
```

## 3. Các Thành Phần Chính

### 3.1. Xử Lý Ảnh (image_processor.py)

- Chuẩn hóa kích thước ảnh (640x640)
- Chuyển đổi ảnh thành tensor
- Xử lý batch và single image

### 3.2. Trích Xuất Đặc Trưng (feature_extractor.py)

- Sử dụng ResNet50 pre-trained
- Chuyển đổi ảnh thành vector đặc trưng
- Hỗ trợ GPU nếu có

### 3.3. Lưu Trữ Vector (vector_store.py)

- Sử dụng ChromaDB
- Lưu trữ và tìm kiếm vector
- Hỗ trợ cosine similarity

### 3.4. Backend (app.py, main.py)

- FastAPI server
- API endpoints:
  - /search: Tìm kiếm ảnh tương tự
  - /images: Lấy danh sách ảnh

### 3.5. Frontend (streamlit_app.py)

- Giao diện người dùng Streamlit
- Upload ảnh
- Hiển thị kết quả tìm kiếm
- Hiển thị độ tương đồng

### 3.6. Đánh Giá Hệ Thống (evaluation.py)

- Tính toán precision
- Tính toán recall
- Tính toán mean average precision (mAP)

## 4. Công Nghệ Sử Dụng

- Python 3.8+
- PyTorch 2.1.0
- FastAPI 0.104.1
- ChromaDB 0.4.18
- Streamlit 1.31.0
- scikit-learn 1.3.2

## 5. Quy Trình Hoạt Động

1. Khởi tạo database (init_db.py)
2. Upload ảnh qua giao diện
3. Trích xuất đặc trưng
4. Tìm kiếm ảnh tương tự
5. Hiển thị kết quả

## 6. Bảo Mật

- License độc quyền
- Bảo vệ bản quyền phần mềm
- Hạn chế sao chép và sửa đổi

## 7. Logging

- Ghi log cho mọi hoạt động
- Theo dõi lỗi và thông tin
- Debug information trong giao diện

## 8. Yêu Cầu Hệ Thống

- Python 3.8+
- Các thư viện trong requirements.txt
- GPU (tùy chọn) cho xử lý nhanh hơn
