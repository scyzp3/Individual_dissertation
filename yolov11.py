import warnings
from ultralytics.models.yolo import model
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model='/home/scyzp3/yolov11/ultralytics/ultralytics/cfg/models/11/yolo11n.pt')
    model.train(data=r'bdd.yaml',
                imgsz=416,
                epochs=50,
                batch=2,
                workers=0,
                device='0',
                optimizer='Adam',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
