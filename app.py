import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
import logging

from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor
from vector_store import VectorStore

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Flower Search System</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .upload-form {
                    border: 2px dashed #ccc;
                    padding: 20px;
                    text-align: center;
                    margin: 20px 0;
                }
                .results {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-top: 20px;
                }
                .result-image {
                    width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <h1>Flower Search System</h1>
            <div class="upload-form">
                <form action="/search" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <button type="submit">Search Similar Flowers</button>
                </form>
            </div>
            <div id="results" class="results"></div>
        </body>
    </html>
    """

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = f"static/uploads/{file.filename}"
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 