import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1️⃣ 加载 YOLO 模型
model = YOLO("runs/train/exp_optimized5/weights/best.pt")  # 你的YOLO模型路径

# 2️⃣ 初始化 DeepSORT 跟踪器（优化参数）
tracker = DeepSort(
    max_cosine_distance=0.5,  # 允许更大相似度，减少重复计数
    nn_budget=200,            # 允许更大的 ReID 模型
    max_age=50,               # 允许目标消失 50 帧后仍恢复 ID
    n_init=5,                 # 需要 5 帧连续检测后确认目标（减少误检）
    embedder="mobilenet",     # 使用 MobileNet 作为特征提取器
    embedder_gpu=True         # 启用 GPU 加速
)

# 3️⃣ 读取视频文件
video_path = "test4.mp4"
cap = cv2.VideoCapture(video_path)

# 4️⃣ 初始化视频写入
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# 5️⃣ YOLO 检测转换为 DeepSORT 格式
def yolo_to_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 目标框坐标
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0].item()  # 置信度
            cls = int(box.cls[0])  # 类别索引

            if conf > 0.4:  # 只跟踪置信度 > 0.4 的目标
                detections.append(([x1, y1, w, h], conf, cls))

    return detections

# 6️⃣ 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🟢 (a) 运行 YOLO 检测
    results = model.predict(frame, conf=0.4)  # 置信度阈值提高到 0.4
    dets_list = yolo_to_detections(results)

    # 🔵 (b) 更新 DeepSORT 目标跟踪
    tracks = tracker.update_tracks(dets_list, frame=frame) if dets_list else []

    # 🔴 (c) 可视化跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x, y, w, h = map(int, track.to_ltwh())  # 获取跟踪目标的坐标

        # 绘制目标框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🟡 (d) 写入处理后的视频帧
    out.write(frame)

# 7️⃣ 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("🎉 处理完成，视频已保存为 output.mp4")
