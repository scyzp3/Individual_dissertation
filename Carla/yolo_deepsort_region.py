
"""Script to perform real-time object detection and tracking using YOLOv11 and DeepSORT."""

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# initialize YOLOv11 model
model = YOLO('../yolov11.pt').to('cuda')

# initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# define screen region for detection
region = {'left': 100, 'top': 100, 'width': 1000, 'height': 600}

# loop for real-time detection
while True:
    start_time = time.time()

    # screen capture
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # YOLO detection
    results = model(img)
    detections = []
    for result in results:
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls_id)))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=img)

    # draw detection results
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the image
    cv2.imshow('Screen Detection with DeepSORT', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()