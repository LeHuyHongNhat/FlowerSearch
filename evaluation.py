import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score
from typing import List, Dict
import os

class SystemEvaluator:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def calculate_precision(self, relevant_images: List[str], retrieved_images: List[str]) -> float:
        """Calculate precision@k."""
        relevant_set = set(relevant_images)
        retrieved_set = set(retrieved_images)
        
        if len(retrieved_set) == 0:
            return 0.0
            
        return len(relevant_set.intersection(retrieved_set)) / len(retrieved_set)

    def calculate_recall(self, relevant_images: List[str], retrieved_images: List[str]) -> float:
        """Calculate recall@k."""
        relevant_set = set(relevant_images)
        retrieved_set = set(retrieved_images)
        
        if len(relevant_set) == 0:
            return 0.0
            
        return len(relevant_set.intersection(retrieved_set)) / len(relevant_set)

    def calculate_map(self, query_results: List[Dict]) -> float:
        """Calculate mean average precision (mAP)."""
        aps = []
        
        for query_result in query_results:
            relevant_images = query_result['relevant_images']
            retrieved_images = [img['image_path'] for img in query_result['retrieved_images']]
            
            # Calculate average precision for this query
            ap = 0.0
            num_relevant = 0
            
            for i, img in enumerate(retrieved_images):
                if img in relevant_images:
                    num_relevant += 1
                    ap += num_relevant / (i + 1)
            
            if len(relevant_images) > 0:
                ap /= len(relevant_images)
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0

    def evaluate_system(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate the system using multiple metrics."""
        metrics = {
            'precision': [],
            'recall': [],
            'map': 0.0
        }
        
        # Calculate precision and recall for each query
        for query in test_queries:
            relevant_images = query['relevant_images']
            retrieved_images = [img['image_path'] for img in query['retrieved_images']]
            
            precision = self.calculate_precision(relevant_images, retrieved_images)
            recall = self.calculate_recall(relevant_images, retrieved_images)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
        
        # Calculate mean precision and recall
        metrics['precision'] = np.mean(metrics['precision'])
        metrics['recall'] = np.mean(metrics['recall'])
        
        # Calculate mAP
        metrics['map'] = self.calculate_map(test_queries)
        
        return metrics

def create_test_queries(dataset_path: str, vector_store) -> List[Dict]:
    """Create test queries from the dataset."""
    test_queries = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        # Get all images in the current directory
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) > 0:
            # Use the first image as a query
            query_image = os.path.join(root, images[0])
            
            # All other images in the same directory are considered relevant
            relevant_images = [os.path.join(root, img) for img in images[1:]]
            
            # Get similar images from the vector store
            # Note: You'll need to process the query image and get its features first
            # This is a placeholder - you'll need to implement the actual feature extraction
            retrieved_images = []  # Placeholder
            
            test_queries.append({
                'query_image': query_image,
                'relevant_images': relevant_images,
                'retrieved_images': retrieved_images
            })
    
    return test_queries 