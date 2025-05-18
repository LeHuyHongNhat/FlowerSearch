import os
from feature_extractor import FeatureExtractor
from vector_store import VectorStore
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_database(image_dir: str = "Dataset"):
    """Initialize the vector database with all images."""
    try:
        # Khởi tạo feature extractor và vector store
        extractor = FeatureExtractor()
        vector_store = VectorStore()

        # Lấy danh sách tất cả các file ảnh
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        logger.info(f"Tìm thấy {len(image_files)} file ảnh")

        # Xử lý từng ảnh và thêm vào vector store
        features_dict = {}
        for image_path in image_files:
            try:
                features = extractor.extract_features(image_path)
                features_dict[image_path] = features
                logger.info(f"Đã xử lý: {image_path}")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {image_path}: {str(e)}")

        # Thêm tất cả features vào vector store
        if features_dict:
            logger.info(f"Đang thêm {len(features_dict)} ảnh vào vector store...")
            vector_store.add_images(features_dict)
            
            # Kiểm tra kết quả
            stats = vector_store.get_collection_stats()
            logger.info(f"Thống kê collection:")
            logger.info(f"- Tổng số items: {stats['total_items']}")
            logger.info(f"- Mẫu IDs: {stats['sample_ids']}")
            logger.info(f"- Mẫu đường dẫn: {stats['sample_paths']}")
        else:
            logger.warning("Không có ảnh nào được xử lý thành công")

    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo database: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_database() 