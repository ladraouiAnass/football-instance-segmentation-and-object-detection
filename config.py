"""
Configuration settings for Football Detection System
"""

from pathlib import Path

# Model Configuration
MODEL_PATH = Path('train2/weights/best.pt')
BASE_MODEL_PATH = Path('yolov8n.pt')
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# File Paths
DEMO_VIDEO = 'demo_video.mp4'
SAMPLE_IMAGE = 'sample_image.png'
OUTPUT_VIDEO = 'output.avi'
OUTPUT_IMAGE = 'output.png'

# Detection Classes
FOOTBALL_CLASSES = {
    0: 'player',
    1: 'ball', 
    2: 'goalkeeper',
    3: 'referee'
}

# Colors for visualization (BGR format)
DETECTION_COLORS = {
    'bbox': (0, 255, 0),        # Green for bounding boxes
    'text': (0, 255, 0),        # Green for text
    'masks': [                  # Colors for segmentation masks
        (0, 0, 255),           # Red
        (255, 0, 0),           # Blue  
        (0, 255, 255),         # Yellow
        (255, 255, 0),         # Cyan
        (255, 0, 255)          # Magenta
    ]
}

# Video Processing Settings
VIDEO_SETTINGS = {
    'codec': 'XVID',
    'skip_frames': 1,
    'resize_factor': 1.0,
    'mask_alpha': 0.4
}

# Performance Settings
PERFORMANCE = {
    'use_gpu': True,
    'batch_size': 1,
    'verbose': False
}