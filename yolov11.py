import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO("yolo11x.pt")  # 选择适合的预训练模型

    model.train(
        data='data2.yaml',
        imgsz=768,             # 适中分辨率，兼顾精度和显存占用
        epochs=50,            # 训练更久
        batch=1,               # 小 batch 适配 4GB 显存
        workers=2,             # 适当使用 worker 加快数据加载
        device='0',
        optimizer='AdamW',     # 选择更好的优化器
        lr0=0.001, lrf=0.1,    # 适配学习率
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,       # 仅前 10 轮使用 Mosaic
        cache='ram',           # 缓存数据到 RAM 加速训练
        auto_augment=True,     # 使用自动增强
        mixup=0.2,             # 启用混合增强
        label_smoothing=0.1,   # 采用标签平滑
        half=True,             # 使用 FP16 训练，减少显存占用
        project='runs/train',
        name='exp_optimized',
    )
