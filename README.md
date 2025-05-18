# Flower Search System

A content-based image retrieval system for flowers using deep learning and vector similarity search.

## Features

- Image preprocessing to standardize image sizes (640x640)
- Feature extraction using pre-trained ResNet50 CNN
- Vector similarity search using ChromaDB
- Web interface for image upload and search
- System evaluation metrics (precision, recall, mAP)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd FlowerSearch
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the System

### Step 1: Prepare the Dataset

Place your flower images in the `Dataset` directory. Each subdirectory should contain images of the same flower type (less than 20 images per directory).

### Step 2: Initialize the Database

Before running the system, you need to initialize the database with your flower images:

```bash
python init_db.py
```

This script will:

- Process all images in the Dataset directory
- Extract features using ResNet50
- Store the features in ChromaDB
- Show progress and any errors during processing

### Step 3: Start the Backend Server

Open a terminal and run:

```bash
python app.py
```

The FastAPI server will start at http://localhost:8000

### Step 4: Start the Frontend Interface

Open another terminal and run:

```bash
streamlit run streamlit_app.py
```

The Streamlit interface will be available at http://localhost:8501

### Step 5: Using the System

1. Open your web browser and go to http://localhost:8501
2. Upload a flower image using the interface
3. Click the "Tìm kiếm" (Search) button
4. View the top 3 most similar images with their similarity scores

## System Components

- `image_processor.py`: Handles image preprocessing and standardization
- `feature_extractor.py`: Extracts features using pre-trained CNN
- `vector_store.py`: Manages vector storage and similarity search
- `app.py`: FastAPI backend server
- `streamlit_app.py`: Streamlit frontend interface
- `evaluation.py`: System evaluation metrics
- `init_db.py`: Database initialization script

## Evaluation

To evaluate the system performance:

```python
from evaluation import SystemEvaluator, create_test_queries
from vector_store import VectorStore

# Initialize components
vector_store = VectorStore()
evaluator = SystemEvaluator(vector_store)

# Create test queries
test_queries = create_test_queries("path/to/dataset", vector_store)

# Evaluate system
metrics = evaluator.evaluate_system(test_queries)
print(metrics)
```

## Troubleshooting

1. If you get a connection error:

   - Make sure the FastAPI backend is running (Step 3)
   - Check if the backend is accessible at http://localhost:8000

2. If images are not displaying:

   - Check if the image paths are correct
   - Ensure the images are in supported formats (jpg, jpeg, png)
   - Make sure you've initialized the database (Step 2)

3. If no similar images are found:

   - Check if the database was properly initialized
   - Verify that your Dataset directory contains images
   - Check the logs for any errors during database initialization

4. If the system is slow:
   - Consider using GPU if available
   - Reduce the number of images in the dataset
   - Optimize the vector search parameters

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- ChromaDB
- Pillow
- scikit-learn
- numpy
- Streamlit
- requests

## License

MIT License
