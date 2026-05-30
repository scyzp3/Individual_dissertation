# 面向自动驾驶的目标检测与跟踪系统

本项目构建了一个面向自动驾驶场景的多模态感知系统，结合 LiDAR 检测、视觉目标检测与多目标跟踪，实现基于 SFA3D、YOLO 和 DeepSORT 的目标检测与融合实验。

## 系统展示

![系统结构](assets/architecture.png)

<video src="assets/demo.mp4" controls width="720"></video>

![实验结果 1](assets/eval1.png)

![实验结果 2](assets/eval2.png)

## 项目结构

```text
.
|-- assets/                  # 图片与演示视频
|-- Carla/                   # CARLA 仿真、传感器与融合实验脚本
|   |-- demos/
|   |-- pipelines/
|   |-- tools/
|   `-- sep/
`-- Yolo_deepsort_training/  # YOLO/DeepSORT 训练与检测脚本
```

## 模块说明

- `Carla/`：包含 CARLA 0.9.12 环境下的数据采集、LiDAR/Camera 测试、SFA3D 检测与 YOLO 融合实验。
- `Yolo_deepsort_training/`：包含 YOLO 训练、标注格式转换、视频检测和 DeepSORT 跟踪相关脚本。
- `assets/`：保存系统结构图、实验结果图和演示视频。

## 备注

- CARLA 相关脚本需要配置 CARLA Python API。
- SFA3D 实验默认使用 KITTI 格式数据。
- 训练依赖可参考 `Yolo_deepsort_training/requirements.txt`。
