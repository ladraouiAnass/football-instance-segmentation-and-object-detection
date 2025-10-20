"""
Football Instance Segmentation for Images
Process single images with detection and segmentation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballImageDetector:
    """Football detection and segmentation for static images"""
    
    def __init__(self, model_path: str = 'train2/weights/best.pt', confidence: float = 0.25):
        """Initialize detector for image processing"""
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self.colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info("Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")
    
    def process_image(self, image_path: str, output_path: str = 'output.png'):
        """Process single image with detection and segmentation"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        logger.info("Processing image...")
        results = self.model.predict(source=image, imgsz=640, conf=self.confidence, verbose=False)
        
        # Draw bounding boxes
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                color = (0, 255, 0)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Apply segmentation masks
        if results[0].masks is not None:
            image = self._apply_masks(image, results[0].masks.xy)
        
        # Save result
        cv2.imwrite(output_path, image)
        logger.info(f"Output saved to {output_path}")
        
        return image
    
    def _apply_masks(self, image, masks):
        """Apply segmentation masks to image"""
        height, width = image.shape[:2]
        
        for i, mask in enumerate(masks):
            mask_array = np.array(mask).reshape(-1, 2).astype(np.int32)
            mask_img = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_img, [mask_array], 255)
            
            # Apply colored overlay
            color = self.colors[i % len(self.colors)]
            mask_colored = np.zeros_like(image)
            mask_colored[mask_img > 0] = color
            
            # Blend with original image
            alpha = 0.3
            image = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)
        
        return image

def main():
    """Main function for image detection"""
    detector = FootballImageDetector()
    detector.process_image('sample_image.png')

if __name__ == "__main__":
    main()