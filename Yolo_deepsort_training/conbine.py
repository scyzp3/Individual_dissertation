"""Experimentally replace a YOLO backbone with a trained ResNet feature extractor."""

import torch
from torchvision import models
from ultralytics import YOLO


def load_resnet_feature_extractor(model_path):
    """Load the classifier weights and return ResNet without its final head."""
    resnet18 = models.resnet18(pretrained=False)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 5)

    resnet18.load_state_dict(torch.load(model_path))
    feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
    return feature_extractor


def integrate_with_yolo(feature_extractor):
    """Attach the ResNet feature extractor to a YOLO model for backbone experiments."""
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.model.backbone = feature_extractor
    return yolo_model


def main():
    classification_model_path = "datasets/classification_model.pth"
    feature_extractor = load_resnet_feature_extractor(classification_model_path)
    yolo_model = integrate_with_yolo(feature_extractor)

    yolo_model.train(
        data="data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0
    )

    yolo_model.save("yolov8_with_resnet_backbone.pt")
    print("Training complete. Model saved as 'yolov8_with_resnet_backbone.pt'.")


if __name__ == "__main__":
    main()
