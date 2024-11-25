# YOLO Model Training and Evaluation Report

## 1. Model Overview
The YOLO model has been trained for object detection on a custom dataset. The model used is **YOLOv8n** (small version), optimized for faster inference and smaller memory usage. The dataset consists of images with several object categories including vehicles, traffic lights, traffic signs, bikes, and motorbikes.

## 2. Training Parameters
- **Epochs**: 50
- **Batch size**: 16
- **Learning rate**: 0.01
- **Optimizer**: SGD
- **Pretrained weights**: Used

## 3. Dataset Overview
The dataset is composed of images from various sources and contains several object classes, as detailed below:

- **Vehicle**: 135 images
- **Traffic light**: 173 images
- **Traffic sign**: 10 images
- **Bike**: 23 images
- **Motorbike**: 10 images

The dataset was divided into training and testing sets, and the model was trained to classify and localize objects in the images.

## 4. Validation Results
After training, the model was evaluated on the validation set. Below are the results for different metrics:

### 4.1 **Overall Performance (all classes)**

- **Precision (P)**: 0.819
- **Recall (R)**: 0.623
- **mAP50**: 0.692
- **mAP50-95**: 0.473

### 4.2 **Class-wise Performance**

| Class         | Precision | Recall | mAP50 | mAP50-95 |
|---------------|-----------|--------|-------|----------|
| Vehicle       | 0.942     | 0.926  | 0.977 | 0.824    |
| Traffic light | 0.875     | 0.457  | 0.687 | 0.305    |
| Traffic sign  | 0.94      | 0.4    | 0.426 | 0.26     |
| Bike          | 0.634     | 0.833  | 0.869 | 0.591    |
| Motorbike     | 0.705     | 0.5    | 0.501 | 0.384    |

### 4.3 **Inference Speed**
- **Preprocess time**: 0.1ms
- **Inference time**: 1.7ms per image
- **Postprocess time**: 1.8ms per image

## 5. Issues Identified
- **Class imbalance**: The dataset has fewer samples for some classes (e.g., traffic signs, motorbikes), which can impact the model's ability to learn accurate representations for these classes.
- **Insufficient data for small classes**: The limited number of examples for traffic signs and motorbikes likely contributed to the model's lower performance on these classes.
- **Lower recall for traffic lights**: The traffic light class had a relatively low recall, which could be improved by augmenting the dataset or using a different detection approach for small or less distinct objects.

## 6. Conclusion
Overall, the model shows good performance on larger classes, especially vehicles. However, there is room for improvement on smaller and less-represented classes. With additional data augmentation and fine-tuning, the model's performance can be further improved, particularly for traffic lights, traffic signs, and motorbikes.

The validation results have been saved to the directory: `runs/detect/train8`.
