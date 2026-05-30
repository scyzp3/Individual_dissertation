import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # High-memory training profile tuned for an RTX 4090-class GPU.
    model = YOLO('./yolo11x.pt')

    model.train(
        data='bdd.yaml',
        imgsz=640,
        epochs=100,
        batch=16,
        workers=8,
        device='0',
        optimizer='AdamW',
        close_mosaic=10,
        resume=False,
        project='runs/train',
        name='yolo11_exp',
        single_cls=False,
        cache='ram',
        amp=True,
        cos_lr=True
    )
