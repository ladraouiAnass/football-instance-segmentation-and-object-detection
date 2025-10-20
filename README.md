# Football Instance Segmentation and Object Detection

Real-time instance segmentation and object detection for football matches, capable of detecting and segmenting players, ball, goalkeeper, referee, and other football-related objects in live video streams.

## 🎯 Features

- **Real-time Detection**: Process live video streams with minimal latency
- **Multi-class Detection**: Identifies players, ball, goalkeeper, referee
- **Instance Segmentation**: Precise pixel-level segmentation masks
- **YOLOv8 Integration**: Leverages state-of-the-art YOLO architecture
- **GPU Acceleration**: CUDA support for enhanced performance

## 📸 Preview

### Detection Results
![Example 1](preview/example1.png)
*Football match with detected players and objects*

![Example 2](preview/example2.png)
*Instance segmentation showing precise player boundaries*

![Example 3](preview/example3.png)
*Multi-object detection in action*

## 🎥 Demo Video

Check out the real-time detection in action:
- **Demo Video**: `demo_video.mp4` - Shows live instance segmentation and object detection on football footage

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage

**Basic Video Detection:**
```python
python football_detector.py
```

**Image Processing:**
```python
python image_detector.py
```

**Advanced Video Segmentation:**
```python
python video_segmentation.py
```

## 📁 Project Structure

```
├── preview/                    # Example detection results
│   ├── example1.png
│   ├── example2.png
│   └── example3.png
├── train2/                     # Training artifacts
│   └── weights/
│       ├── best.pt            # Best trained model
│       └── last.pt            # Latest checkpoint
├── football_detector.py        # Main video detection script
├── image_detector.py          # Single image processing
├── video_segmentation.py      # Advanced video segmentation
├── demo_video.mp4             # Demo input video
├── sample_image.png           # Sample test image
├── requirements.txt           # Python dependencies
└── yolov8n.pt                # Base YOLO model
```

## 🔧 Model Details

- **Architecture**: YOLOv8 with instance segmentation
- **Input Size**: 640x640
- **Confidence Threshold**: 0.25
- **Classes**: Players, Ball, Goalkeeper, Referee
- **Trained Weights**: `train2/weights/best.pt`

## 🎮 Controls

- Press `q` to quit the video processing
- Output is saved as `output.avi`

## 📊 Performance

The model achieves real-time performance on modern GPUs with accurate detection and segmentation of football-related objects.

## 🤝 Contributing

Feel free to contribute by improving the model accuracy, adding new classes, or optimizing performance.

## 📄 License

This project is open source and available under standard licensing terms.
