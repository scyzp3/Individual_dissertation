import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Small-batch profile for lower-memory GPUs.
    model = YOLO("yolo12n.pt")

    model.train(
        data='data2.yaml',
        imgsz=768,
        epochs=50,
        batch=1,
        workers=2,
        device='0',
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        cache='ram',
        auto_augment=True,
        mixup=0.2,
        label_smoothing=0.1,
        half=True,
        project='runs/train',
        name='exp_optimized',
    )
