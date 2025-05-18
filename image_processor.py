import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path):
        """Process a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def process_directory(self, directory_path):
        """Process all images in a directory."""
        processed_images = {}
        image_paths = []

        # Walk through all subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image_paths.append(image_path)

        # Process images with progress bar
        for image_path in tqdm(image_paths, desc="Processing images"):
            processed_image = self.process_image(image_path)
            if processed_image is not None:
                processed_images[image_path] = processed_image

        return processed_images

    def process_single_image(self, image_path):
        """Process a single image and return the tensor."""
        return self.process_image(image_path) 