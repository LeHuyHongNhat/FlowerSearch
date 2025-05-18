import chromadb
import numpy as np
import os
from typing import List, Dict, Tuple
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="flower_images",
            metadata={"hnsw:space": "cosine"}
        )

    def add_images(self, features_dict: Dict[str, np.ndarray]):
        """Add image features to the vector store."""
        try:
            ids = []
            embeddings = []
            metadatas = []

            for image_path, features in features_dict.items():
                # Tạo ID duy nhất từ đường dẫn đầy đủ
                image_id = str(hash(image_path))
                
                # Convert numpy array to list for ChromaDB
                embedding = features.tolist()
                
                # Create metadata with the full image path
                metadata = {"image_path": image_path}
                
                ids.append(image_id)
                embeddings.append(embedding)
                metadatas.append(metadata)

            # Log thông tin trước khi thêm vào collection
            logger.info(f"Đang thêm {len(ids)} items vào collection")
            logger.info(f"ID đầu tiên: {ids[0]}")
            logger.info(f"Đường dẫn đầu tiên: {metadatas[0]['image_path']}")

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

            # Kiểm tra số lượng items sau khi thêm
            collection_count = len(self.collection.get()['ids'])
            logger.info(f"Số lượng items trong collection sau khi thêm: {collection_count}")

        except Exception as e:
            logger.error(f"Lỗi khi thêm images vào vector store: {str(e)}")
            raise

    def search_similar_images(self, query_features: np.ndarray, n_results: int = 3) -> List[Dict]:
        """Search for similar images based on feature vector."""
        try:
            # Convert query features to list
            query_embedding = query_features.tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            similar_images = []
            for i in range(len(results['ids'][0])):
                similar_images.append({
                    'image_path': results['metadatas'][0][i]['image_path'],
                    'distance': results['distances'][0][i]
                })
            
            return similar_images
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm similar images: {str(e)}")
            return []

    def get_all_images(self) -> List[str]:
        """Get all image paths from the vector store."""
        try:
            results = self.collection.get()
            return [metadata['image_path'] for metadata in results['metadatas']]
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách images: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            results = self.collection.get()
            return {
                'total_items': len(results['ids']),
                'sample_ids': results['ids'][:5],
                'sample_paths': [metadata['image_path'] for metadata in results['metadatas'][:5]]
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy collection stats: {str(e)}")
            return {'error': str(e)} 