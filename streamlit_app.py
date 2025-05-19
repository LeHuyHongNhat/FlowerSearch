import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
from pathlib import Path
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Flower Search System",
    page_icon="🌸",
    layout="wide"
)

# Tiêu đề và mô tả
st.title("🌸 Flower Search System")
st.markdown("""
Tìm kiếm hoa tương tự dựa trên hình ảnh. Hệ thống sẽ trả về 3 hình ảnh có độ tương đồng cao nhất.
""")

# Upload section
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Chọn một hình ảnh hoa...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã upload
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)  # Giới hạn chiều rộng 300px
    
    # Nút tìm kiếm
    if st.button("Tìm kiếm"):
        # Chuẩn bị file để gửi đến API
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        
        try:
            # Gửi request đến FastAPI backend
            response = requests.post("http://localhost:8000/search", files=files)
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Response data: {results}")
                
                if results.get("status") == "success":
                    st.subheader("Kết quả tìm kiếm")
                    
                    # Tạo 3 cột để hiển thị kết quả
                    cols = st.columns(3)
                    
                    # Hiển thị 3 hình ảnh tương tự
                    for i, (col, similar_image) in enumerate(zip(cols, results["similar_images"])):
                        with col:
                            try:
                                image_path = similar_image["image_path"]
                                logger.info(f"Trying to load image: {image_path}")
                                
                                # Kiểm tra xem file có tồn tại không
                                if os.path.exists(image_path):
                                    img = Image.open(image_path)
                                    st.image(img, width=250)  # Giới hạn chiều rộng 250px
                                    
                                    # Hiển thị độ tương đồng
                                    similarity = 1 - similar_image["distance"]
                                    st.markdown(f"**Độ tương đồng:** {similarity:.2%}")
                                    
                                    # Hiển thị thông tin file chi tiết hơn
                                    file_path = Path(image_path)
                                    st.markdown(f"""
                                    **Thông tin file:**
                                    - Tên file: {file_path.name}
                                    - Thư mục: {file_path.parent.name}
                                    - Đường dẫn đầy đủ: {image_path}
                                    """)
                                else:
                                    st.error(f"Không tìm thấy file: {image_path}")
                                    logger.error(f"File not found: {image_path}")
                            except Exception as e:
                                st.error(f"Lỗi khi hiển thị hình ảnh: {str(e)}")
                                logger.error(f"Error displaying image: {str(e)}")
                else:
                    st.error(f"Lỗi từ server: {results.get('message', 'Unknown error')}")
            else:
                st.error(f"Có lỗi xảy ra khi tìm kiếm. Status code: {response.status_code}")
                st.error(f"Response: {response.text}")
        except Exception as e:
            st.error(f"Lỗi kết nối đến server: {str(e)}")
            logger.error(f"Connection error: {str(e)}")
            st.info("Hãy đảm bảo FastAPI server đang chạy tại http://localhost:8000")

# Thêm hướng dẫn sử dụng
with st.expander("Hướng dẫn sử dụng"):
    st.markdown("""
    1. Upload một hình ảnh hoa bằng cách kéo thả hoặc chọn file
    2. Nhấn nút 'Tìm kiếm' để tìm các hình ảnh tương tự
    3. Hệ thống sẽ hiển thị 3 hình ảnh có độ tương đồng cao nhất
    4. Mỗi kết quả sẽ hiển thị độ tương đồng với hình ảnh gốc
    """)

# Thêm thông tin về hệ thống
with st.expander("Thông tin về hệ thống"):
    st.markdown("""
    Hệ thống sử dụng:
    - ResNet50 để trích xuất đặc trưng hình ảnh
    - ChromaDB để lưu trữ và tìm kiếm vector
    - FastAPI làm backend
    - Streamlit làm frontend
    """)

# Thêm phần debug
with st.expander("Debug Information"):
    st.markdown("### Current Working Directory")
    st.code(os.getcwd())
    
    st.markdown("### Static Directory Contents")
    static_dir = "static"
    if os.path.exists(static_dir):
        st.code("\n".join(os.listdir(static_dir)))
    else:
        st.error(f"Static directory not found: {static_dir}")
    
    st.markdown("### Uploads Directory Contents")
    uploads_dir = os.path.join(static_dir, "uploads")
    if os.path.exists(uploads_dir):
        st.code("\n".join(os.listdir(uploads_dir)))
    else:
        st.error(f"Uploads directory not found: {uploads_dir}")
    
    # Thêm thông tin về Dataset directory
    st.markdown("### Dataset Directory Contents")
    dataset_dir = "Dataset"
    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir):
            st.markdown(f"#### Directory: {root}")
            st.code("\n".join(files))
    else:
        st.error(f"Dataset directory not found: {dataset_dir}") 