import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Define paths to the model and dataset
model_path = 'runs/segment/train2/weights/best.pt'  # Path to your saved model
video_path = 'video.mp4'  # Path to your input video

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLO model for instance segmentation and move it to the GPU if available
print("Initializing model...")
model = YOLO(model_path).to(device)

# Open the video file
print("Opening video...")
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
print("Getting video properties...")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

print("Start processing...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to process fewer frames per second
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 360))

    # Perform instance segmentation on the resized frame
    results = model.predict(source=resized_frame, imgsz=640, conf=0.25, device=device)

    # Draw the results on the frame
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1 * frame_width / 640), int(y1 * frame_height / 360), int(x2 * frame_width / 640), int(y2 * frame_height / 360)
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put label near the bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check if masks are present
    if results[0].masks is not None:
        # Get masks and colors for each instance
        masks = results[0].masks.xy
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]  # Example colors, you can adjust as needed

        # Resize the masks to match the dimensions of the original frame
        original_height, original_width = frame.shape[:2]
        resized_masks = np.zeros((len(masks), original_height, original_width), dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask = np.array(mask)
            mask = mask.reshape(-1, 2).astype(np.int32)
            resized_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            cv2.fillPoly(resized_mask, [mask], color=(255, 255, 255))
            resized_masks[i] = resized_mask

        # Apply masks to
