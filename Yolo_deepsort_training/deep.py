
"""Script to perform object detection and tracking using YOLOv11 and DeepSORT."""

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# load YOLOv11 model
model = YOLO("runs/train/exp_optimized5/weights/best.pt")

# deepsort tracker
tracker = DeepSort(
    max_cosine_distance=0.5,
    nn_budget=200,
    max_age=50,
    n_init=5,
    embedder="mobilenet",
    embedder_gpu=True
)

# load video
video_path = "test4.mp4"
cap = cv2.VideoCapture(video_path)

# original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# convert YOLO results to detections
def yolo_to_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # detection box coordinates
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0].item()  # get confidence score
            cls = int(box.cls[0])  # get class id

            if conf > 0.4:  # only keep detections with confidence > 0.4
                detections.append(([x1, y1, w, h], conf, cls))

    return detections

# process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # yolo detection
    results = model.predict(frame, conf=0.4)  # 置信度阈值提高到 0.4
    dets_list = yolo_to_detections(results)

    # deepsort tracking
    tracks = tracker.update_tracks(dets_list, frame=frame) if dets_list else []

    # visualize detections and tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x, y, w, h = map(int, track.to_ltwh())  # get bounding box coordinates

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # write frame to output video
    out.write(frame)

# deallocate resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("processing done")
