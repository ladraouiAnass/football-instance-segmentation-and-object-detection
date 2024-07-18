import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'runs/segment/train2/weights/best.pt' 
video_path = 'video3.mp4'  
print("Initializing model...")
model = YOLO(model_path)

print("Opening video...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Getting video properties...")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

print("Start processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.25)

    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if results[0].masks is not None:
        masks = results[0].masks.xy
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255)] 

        original_height, original_width = frame.shape[:2]
        resized_masks = np.zeros((len(masks), original_height, original_width), dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask = np.array(mask)
            mask = mask.reshape(-1, 2).astype(np.int32)
            resized_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            cv2.fillPoly(resized_mask, [mask], color=(255, 255, 255))
            resized_masks[i] = resized_mask

        for i, mask in enumerate(resized_masks):
            mask = cv2.resize(mask, (frame_width, frame_height))
            mask = np.expand_dims(mask, axis=-1)
            color = colors[i % len(colors)]
            frame = np.where(mask > 0, color, frame)

    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    out.write(frame)

    cv2.imshow('Instance Segmentation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
