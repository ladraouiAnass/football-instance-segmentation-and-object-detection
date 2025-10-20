"""
Football Instance Segmentation and Object Detection
Main detection module for processing video streams
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDetector:
    """Football object detection and instance segmentation using YOLOv8"""
    
    def __init__(self, model_path: str = 'train2/weights/best.pt', confidence: float = 0.25):
        """Initialize the detector with model and confidence threshold"""
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info("Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")
    
    def process_video(self, video_path: str, output_path: str = 'output.avi'):
        """Process video file and save results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        logger.info("Processing video...")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_frame = self._detect_objects(frame)
                
                out.write(processed_frame)
                cv2.imshow('Football Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {frame_count} frames. Output saved to {output_path}")
    
    def _detect_objects(self, frame):
        """Detect objects in a single frame"""
        results = self.model.predict(source=frame, imgsz=640, conf=self.confidence, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame

def main():
    """Main function to run football detection"""
    detector = FootballDetector()
    detector.process_video('demo_video.mp4')

if __name__ == "__main__":
    main()