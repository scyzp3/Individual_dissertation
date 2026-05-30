
"""Script to perform real-time object detection using YOLOv11 on a specified screen region."""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time

# original YOLOv11
model = YOLO('../yolov11.pt').to('cuda')
region = {'left': 100, 'top': 100, 'width': 1000, 'height': 600}

# loop for real-time detection
while True:
    start_time = time.time()

    # capture screen
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # YOLO detection
    results = model(img)

    # draw detection results
    for result in results:
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the image
    cv2.imshow('Screen Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()