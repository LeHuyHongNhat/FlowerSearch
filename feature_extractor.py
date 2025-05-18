import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model_name='resnet50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Định nghĩa transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_name):
        """Load pre-trained model and remove the last layer."""
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove the last fully connected layer
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Model {model_name} not supported")
        return model

    def _preprocess_image(self, image_path):
        """Preprocess image from path to tensor."""
        try:
            # Đọc ảnh
            image = Image.open(image_path).convert('RGB')
            # Áp dụng transform
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            raise

    def extract_features(self, image_path):
        """Extract features from an image path."""
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            
            with torch.no_grad():
                # Add batch dimension if not present
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                
                # Move tensor to device
                image_tensor = image_tensor.to(self.device)
                
                # Extract features
                features = self.model(image_tensor)
                
                # Convert to numpy array and flatten
                features = features.squeeze().cpu().numpy()
                
                # Normalize features
                features = features / np.linalg.norm(features)
                
                return features
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất đặc trưng từ {image_path}: {str(e)}")
            raise

    def extract_features_batch(self, image_paths):
        """Extract features from a list of image paths."""
        features_dict = {}
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                features_dict[image_path] = features
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {image_path}: {str(e)}")
                continue
        return features_dict 