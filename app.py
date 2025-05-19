import os
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
from config import Config
from logger import logger

from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from vector_store import VectorStore

app = FastAPI(title="Flower Search System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
image_processor = ImageProcessor()
feature_extractor = FeatureExtractor()
vector_store = VectorStore()

# Create static directory for uploaded images
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        logger.info(f"Saved uploaded file to: {file_location}")
        
        # Process image
        image_tensor = image_processor.process_single_image(file_location)
        if image_tensor is None:
            return {"status": "error", "message": "Failed to process image"}
        
        # Extract features
        features = feature_extractor.extract_features(file_location)
        
        # Search similar images
        similar_images = vector_store.search_similar_images(features)
        
        # Log the results
        logger.info(f"Found {len(similar_images)} similar images")
        for img in similar_images:
            logger.info(f"Similar image: {img['image_path']}")
        
        # Return results with absolute paths
        return {
            "status": "success",
            "query_image": os.path.abspath(file_location),
            "similar_images": [
                {
                    "image_path": os.path.abspath(img["image_path"]),
                    "distance": img["distance"]
                }
                for img in similar_images
            ]
        }
    except Exception as e:
        logger.error(f"Error in search_similar_images: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT) 