import torch
from torchvision import models
from ultralytics import YOLO


def load_resnet_feature_extractor(model_path):
    """
    Load a pretrained ResNet model and extract its feature extractor.
    """
    # Load the ResNet model
    resnet18 = models.resnet18(pretrained=False)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 5)  # Adjust output for 5 classes (example)

    # Load trained weights
    resnet18.load_state_dict(torch.load(model_path))

    # Extract feature extractor (all layers except the classification head)
    feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
    return feature_extractor


def integrate_with_yolo(feature_extractor):
    """
    Replace YOLO's backbone with the ResNet feature extractor.
    """
    # Load YOLO model (e.g., YOLOv8n)
    yolo_model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

    # Replace the backbone with the ResNet feature extractor
    yolo_model.model.backbone = feature_extractor

    # Return modified YOLO model
    return yolo_model


def main():
    # Step 1: Load the ResNet feature extractor
    classification_model_path = "datasets/classification_model.pth"  # Path to your classification model
    feature_extractor = load_resnet_feature_extractor(classification_model_path)

    # Step 2: Integrate the feature extractor into YOLO
    yolo_model = integrate_with_yolo(feature_extractor)

    # Step 3: Train the YOLO model
    data_yaml_path = "data.yaml"  # Update with the path to your YOLO dataset YAML file
    yolo_model.train(
        data=data_yaml_path,
        epochs=50,
        batch=16,
        imgsz=640,
        device=0  # Use GPU if available
    )

    # Step 4: Save the trained YOLO model
    yolo_model.save("yolov8_with_resnet_backbone.pt")
    print("Training complete. Model saved as 'yolov8_with_resnet_backbone.pt'.")


if __name__ == "__main__":
    main()
