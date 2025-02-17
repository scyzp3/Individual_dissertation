import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO('./yolo11x.pt')

    model.train(
        data='bdd.yaml',  # 数据集配置文件
        imgsz=640,  # 提高输入图像尺寸（更高精度）
        epochs=100,  # 训练 100 轮
        batch=16,  # 增大 batch size，充分利用 4090 显存
        workers=8,  # 增加数据加载线程，加快训练
        device='0',  # 选择 GPU 设备
        optimizer='AdamW',  # AdamW 比 Adam 更稳定
        close_mosaic=10,  # 关闭 Mosaic 增强（默认10）
        resume=False,  # 不从上次训练恢复
        project='runs/train',  # 训练结果存储路径
        name='yolo11_exp',  # 自定义实验名称
        single_cls=False,  # 是否只检测一个类别
        cache='ram',  # 使用 RAM 加速数据加载（可选）
        amp=True,  # 开启混合精度训练（减少显存占用）
        cos_lr=True  # 使用余弦退火学习率调度
    )
