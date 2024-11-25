# YOLOv8 Training Report

## Overview
This document summarizes the training of a YOLOv8 model for object detection on a custom traffic dataset. The objective was to detect vehicles, traffic lights, traffic signs, bikes, and motorbikes in urban scenarios.

---

## Dataset
- **Total Images**: 779 (train), 249 (validation)
- **Classes**: Vehicle, Traffic Light, Traffic Sign, Bike, Motorbike
- **Image Size**: 640x640
- **Annotations**: YOLO format

---

## Training Configuration
- **Model**: YOLOv8 Nano (`yolov8n.pt`)
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Hardware**: NVIDIA RTX 3050 Ti

---

## Results
| Metric       | Value  |
|--------------|--------|
| Precision    | 0.819  |
| Recall       | 0.623  |
| mAP@0.5      | 0.692  |
| mAP@0.5:0.95 | 0.473  |

**Class Performance**:
- Vehicle: mAP@0.5 = 0.977
- Traffic Light: mAP@0.5 = 0.687
- Traffic Sign: mAP@0.5 = 0.426
- Bike: mAP@0.5 = 0.869
- Motorbike: mAP@0.5 = 0.501

---

## Analysis
### Causes of Performance Issues:
1. **Class Imbalance**: The dataset contains significantly more samples for vehicles compared to traffic signs or motorbikes, leading to underfitting for less-represented classes.
2. **Complex Backgrounds**: Traffic lights and signs often blend into complex urban backgrounds, making detection more challenging.
3. **Small Objects**: Traffic lights and signs are smaller in size relative to vehicles, which may reduce detection accuracy.
4. **Limited Dataset**: The dataset size may not be sufficient to capture the full variability of objects in different environments.

---

## Conclusion
The model achieved strong performance in detecting vehicles but struggled with smaller or underrepresented objects such as traffic lights and signs. Future work will focus on addressing these challenges through dataset improvements and model optimization.
