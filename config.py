class Config:
    # Image processing
    IMAGE_SIZE = (640, 640)
    MODEL_NAME = 'resnet50'
    
    # Vector store
    VECTOR_STORE_PATH = "chroma_db"
    COLLECTION_NAME = "flower_images"
    
    # File paths
    UPLOAD_DIR = "static/uploads"
    DATASET_DIR = "Dataset"
    
    # Search
    NUM_RESULTS = 3
    
    # API
    HOST = "0.0.0.0"
    PORT = 8000 