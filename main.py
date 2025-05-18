from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import shutil
from feature_extractor import FeatureExtractor
from vector_store import VectorStore
import logging
import uuid

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Tạo thư mục static/uploads nếu chưa tồn tại
os.makedirs("static/uploads", exist_ok=True)

# Mount thư mục static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo feature extractor và vector store
feature_extractor = FeatureExtractor()
vector_store = VectorStore()

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    try:
        # Tạo tên file ngẫu nhiên
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("static/uploads", unique_filename)
        
        # Lưu file upload
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to: {file_path}")
        
        # Trích xuất đặc trưng từ ảnh
        features = feature_extractor.extract_features(file_path)
        
        # Tìm kiếm ảnh tương tự
        similar_images = vector_store.search_similar_images(features)
        
        # Chuyển đổi đường dẫn tương đối thành URL
        for img in similar_images:
            img['image_path'] = f"/static/{os.path.relpath(img['image_path'], 'static')}"
        
        return JSONResponse(content={
            "status": "success",
            "similar_images": similar_images
        })
        
    except Exception as e:
        logger.error(f"Error in search_similar_images: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/images")
async def get_all_images():
    try:
        images = vector_store.get_all_images()
        # Chuyển đổi đường dẫn tương đối thành URL
        images = [f"/static/{os.path.relpath(img, 'static')}" for img in images]
        return JSONResponse(content={
            "status": "success",
            "images": images
        })
    except Exception as e:
        logger.error(f"Error in get_all_images: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 