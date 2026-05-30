"""Run YOLO detection with DeepSORT tracking on a video."""

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("runs/train/exp_optimized5/weights/best.pt")

tracker = DeepSort(
    max_cosine_distance=0.5,
    nn_budget=200,
    max_age=50,
    n_init=5,
    embedder="mobilenet",
    embedder_gpu=True
)

video_path = "test4.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def yolo_to_detections(results):
    """Convert YOLO xyxy boxes to DeepSORT xywh detections."""
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if conf > 0.4:
                detections.append(([x1, y1, w, h], conf, cls))

    return detections


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4)
    dets_list = yolo_to_detections(results)

    tracks = tracker.update_tracks(dets_list, frame=frame) if dets_list else []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x, y, w, h = map(int, track.to_ltwh())

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("processing done")
