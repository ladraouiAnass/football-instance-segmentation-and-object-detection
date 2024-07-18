import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'runs/segment/train2/weights/best.pt'
image_path = 'img1.png'
output_path = 'output.png'


print("Initializing model...")
model = YOLO(model_path)
print("Loading image...")
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}.")
    exit()

print("Performing prediction...")
results = model.predict(source=image, imgsz=640, conf=0.25)

for result in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = result
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    label = f"{model.names[int(cls)]}: {conf:.2f}"
    color = (0, 255, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

if results[0].masks is not None:
    masks = results[0].masks.xy
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255)] 

    original_height, original_width = image.shape[:2]
    resized_masks = np.zeros((len(masks), original_height, original_width), dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask = np.array(mask)
        mask = mask.reshape(-1, 2).astype(np.int32)
        resized_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        cv2.fillPoly(resized_mask, [mask], color=(255, 255, 255))
        resized_masks[i] = resized_mask

    for i, mask in enumerate(resized_masks):
        mask = np.expand_dims(mask, axis=-1)
        color = colors[i % len(colors)]
        image = np.where(mask > 0, color, image)

if image.dtype != np.uint8:
    image = cv2.convertScaleAbs(image)


print("Saving output image...")
cv2.imwrite(output_path, image)

print("Process completed. Output saved to", output_path)
