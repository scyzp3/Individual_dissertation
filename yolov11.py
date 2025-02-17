import warnings
from ultralytics.models.yolo import model
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model='/home/scyzp3/yolov11/ultralytics/ultralytics/cfg/models/11/yolo11n.pt')
    model.train(data=r'bdd.yaml',
                imgsz=416,  # 更小的图像尺寸，减少显存压力
                epochs=50,
                batch=2,  # 减少批量大小，避免显存溢出
                workers=0,  # 可以调整为 > 0 来加速数据加载，视硬件而定
                device='0',  # 显式指定使用的 GPU，'0' 代表第一块 GPU
                optimizer='Adam',  # 可以改为 'Adam' 尝试更快收敛
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
