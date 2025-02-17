from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("runs/train/exp5/weights/best.pt")

# 读取视频
video_path = "datasets/bdd100k_videos_train_00/bdd100k/videos/train/00a0f008-3c67908e.mov"
cap = cv2.VideoCapture(video_path)

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度

# 创建输出视频文件
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 运行 YOLO 目标检测
    results = model(frame)

    # 画出检测框
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取检测框坐标
            conf = box.conf[0].item()  # 置信度
            cls = int(box.cls[0])  # 类别索引
            label = f"{model.names[cls]} {conf:.2f}"

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存处理后的视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
