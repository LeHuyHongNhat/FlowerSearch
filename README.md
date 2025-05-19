# Flower Search System 🌸

Hệ thống tìm kiếm ảnh hoa tương tự sử dụng deep learning và vector similarity search.

## Tính Năng Chính

- Tiền xử lý ảnh chuẩn hóa kích thước (640x640)
- Trích xuất đặc trưng sử dụng ResNet50 CNN đã được huấn luyện trước
- Tìm kiếm vector tương tự sử dụng ChromaDB
- Giao diện web cho việc upload và tìm kiếm ảnh
- Đánh giá hệ thống (precision, recall, mAP)

## Yêu Cầu Hệ Thống

- Python 3.8+
- PyTorch 2.2.1
- FastAPI 0.110.0
- ChromaDB 0.4.22
- Streamlit 1.32.0
- Các thư viện khác trong requirements.txt

## Cài Đặt

1. Clone repository:

```bash
git clone https://github.com/yourusername/FlowerSearch.git
cd FlowerSearch
```

2. Tạo môi trường ảo và cài đặt dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Chạy Hệ Thống

### Bước 1: Chuẩn Bị Dataset

Đặt ảnh hoa vào thư mục `Dataset`. Mỗi thư mục con nên chứa ảnh của cùng một loại hoa (ít hơn 20 ảnh mỗi thư mục).

### Bước 2: Khởi Tạo Database

```bash
python init_db.py
```

Script này sẽ:

- Xử lý tất cả ảnh trong thư mục Dataset
- Trích xuất đặc trưng sử dụng ResNet50
- Lưu trữ đặc trưng trong ChromaDB
- Hiển thị tiến trình và lỗi (nếu có)

### Bước 3: Chạy Backend Server

```bash
python app.py
```

FastAPI server sẽ chạy tại http://localhost:8000

### Bước 4: Chạy Frontend Interface

```bash
streamlit run streamlit_app.py
```

Giao diện Streamlit sẽ có sẵn tại http://localhost:8501

## Cấu Trúc Dự Án

```
FlowerSearch/
├── Dataset/              # Chứa ảnh hoa
├── chroma_db/           # Vector database
├── static/
│   └── uploads/        # Thư mục lưu ảnh upload
├── app.py              # FastAPI backend
├── streamlit_app.py    # Frontend interface
├── image_processor.py  # Xử lý ảnh
├── feature_extractor.py # Trích xuất đặc trưng
├── vector_store.py     # Quản lý vector database
├── evaluation.py       # Đánh giá hệ thống
├── init_db.py         # Khởi tạo database
├── config.py          # Cấu hình hệ thống
├── logger.py          # Cấu hình logging
├── requirements.txt   # Dependencies
└── README.md         # Tài liệu hướng dẫn
```

## Sử Dụng

1. Mở trình duyệt và truy cập http://localhost:8501
2. Upload ảnh hoa qua giao diện
3. Nhấn nút "Tìm kiếm"
4. Xem 3 ảnh tương tự nhất với điểm tương đồng

## Đánh Giá Hệ Thống

```python
from evaluation import SystemEvaluator, create_test_queries
from vector_store import VectorStore

# Khởi tạo components
vector_store = VectorStore()
evaluator = SystemEvaluator(vector_store)

# Tạo test queries
test_queries = create_test_queries("path/to/dataset", vector_store)

# Đánh giá hệ thống
metrics = evaluator.evaluate_system(test_queries)
print(metrics)
```

## Xử Lý Sự Cố

1. Lỗi kết nối:

   - Kiểm tra FastAPI backend đang chạy
   - Xác nhận backend có thể truy cập tại http://localhost:8000

2. Ảnh không hiển thị:

   - Kiểm tra đường dẫn ảnh
   - Đảm bảo ảnh đúng định dạng (jpg, jpeg, png)
   - Xác nhận đã khởi tạo database

3. Không tìm thấy ảnh tương tự:

   - Kiểm tra database đã được khởi tạo đúng
   - Xác nhận thư mục Dataset có ảnh
   - Kiểm tra logs để tìm lỗi

4. Hệ thống chạy chậm:
   - Sử dụng GPU nếu có
   - Giảm số lượng ảnh trong dataset
   - Tối ưu tham số tìm kiếm vector

## Giấy Phép

Xem file [LICENSE.md](LICENSE.md) để biết thêm chi tiết.

## Liên Hệ

Le Huy Hong Nhat
Email: nhat050403@gmail.com
