import cv2
import torch
from ultralytics import YOLO
import os

model_path = 'train2/weights/best.pt'
video_path = 'video3.mp4'  
print("init model...")
model = YOLO(model_path)
print("opening video...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print("getting video properties...")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
print("start processing...")
while True:
    print(1)
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
    out.write(frame)
    cv2.imshow('Instance Segmentation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
