import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# 1) Load YOLO model
model = YOLO("runs/train/exp5/weights/best.pt")  # Specify your YOLO weights (local path or from the model hub)

# 2) Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=100,
    n_init=10,
    nn_budget=100,
    max_cosine_distance=0.2,   # ReID embedding distance threshold
    embedder="mobilenet",      # Default embedder model (can be changed)
    embedder_gpu=True,         # Use GPU for embedding
)

# 3) Load the video file
video_path = "datasets/bdd100k_videos_train_00/bdd100k/videos/train/00a0f008-3c67908e.mov"
cap = cv2.VideoCapture(video_path)

# 4) Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = "output_video.mp4"
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# 5) Convert YOLO detections to a format compatible with DeepSORT
def yolo_to_detections(boxes):
    """
    Convert YOLO detections to DeepSORT-compatible format:
    [([x, y, w, h], confidence, class)] (list of tuples)
    """
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])    # YOLO class ID
        conf = float(box.conf[0])   # Confidence score
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Only track 'person' class (class 0 in COCO dataset)
        if cls_id == 0:
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]   # Format: [x, y, w, h]
            detections.append((bbox, conf, cls_id))
    return detections

# 6) Main loop: read and process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (a) Detect objects with YOLO
    results = model.predict(frame, conf=0.5)
    boxes = results[0].boxes

    # (b) Convert YOLO detections to DeepSORT format
    dets_list = yolo_to_detections(boxes)

    # (c) Update tracker with new detections
    tracks = tracker.update_tracks(dets_list, frame=frame) if len(dets_list) > 0 else []

    # (d) Visualize tracking results on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x, y, w, h = map(int, track.to_ltwh())  # Bounding box coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # (e) Write the processed frame to the output video
    out.write(frame)

# 7) Release video capture and writer resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. The output video is saved to:", out_path)