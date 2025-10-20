"""
Advanced Football Video Segmentation
Enhanced video processing with instance segmentation masks
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballVideoSegmentation:
    """Advanced football video processing with segmentation"""
    
    def __init__(self, model_path: str = 'train2/weights/best.pt', confidence: float = 0.25):
        """Initialize video segmentation processor"""
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        self._load_model()
    
    def _load_model(self):
        """Load and configure YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model on {self.device}...")
        self.model = YOLO(str(self.model_path)).to(self.device)
        logger.info("Model loaded successfully")
    
    def process_video(self, video_path: str, output_path: str = 'segmented_output.avi', 
                     skip_frames: int = 1, resize_factor: float = 1.0):
        """Process video with advanced segmentation options"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust dimensions if resizing
        if resize_factor != 1.0:
            frame_width = int(frame_width * resize_factor)
            frame_height = int(frame_height * resize_factor)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        logger.info(f"Processing {total_frames} frames...")
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % skip_frames != 0:
                    continue
                
                # Resize frame if needed
                if resize_factor != 1.0:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                processed_frame = self._process_frame(frame)
                processed_count += 1
                
                out.write(processed_frame)
                cv2.imshow('Football Segmentation', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if processed_count % 30 == 0:
                    logger.info(f"Processed {processed_count} frames...")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            logger.info(f"Processing complete. Output saved to {output_path}")
    
    def _process_frame(self, frame):
        """Process single frame with detection and segmentation"""
        results = self.model.predict(source=frame, imgsz=640, conf=self.confidence, 
                                   device=self.device, verbose=False)
        
        # Draw bounding boxes
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Apply segmentation masks
        if results[0].masks is not None:
            frame = self._apply_segmentation_masks(frame, results[0].masks.xy)
        
        return frame
    
    def _apply_segmentation_masks(self, frame, masks):
        """Apply colored segmentation masks with transparency"""
        height, width = frame.shape[:2]
        
        for i, mask in enumerate(masks):
            mask_points = np.array(mask).reshape(-1, 2).astype(np.int32)
            
            # Create mask
            mask_img = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_img, [mask_points], 255)
            
            # Apply colored overlay
            color = self.colors[i % len(self.colors)]
            overlay = frame.copy()
            overlay[mask_img > 0] = color
            
            # Blend with original frame
            alpha = 0.4
            frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        return frame

def main():
    """Main function for video segmentation"""
    processor = FootballVideoSegmentation()
    processor.process_video('demo_video.mp4', skip_frames=2, resize_factor=0.8)

if __name__ == "__main__":
    main()