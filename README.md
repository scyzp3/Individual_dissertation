# Object Detection and Tracking System for Autonomous Vehicles

## Introduction
This study aims to develop an efficient and adaptive multimodal fusion framework that integrates LiDAR (SFA3D) and visual (YOLOv11) detection through dynamic IoU threshold adjustment and confidence-weighted fusion. 

## Strcture
The repository is organized with mainly two folders

### Yolo_deepsort_training
The first folder contains the training code for YOLOv11 and DeepSort. The training process is based on the YOLOv11 model, which is a state-of-the-art object detection model. The DeepSort algorithm is used for tracking objects across frames.

### Carla
The second folder contains the code for the CARLA simulator(Version 0.9.12). The CARLA simulator is an open-source autonomous driving simulator that provides a realistic environment for testing and evaluating autonomous driving algorithms. The code in this folder includes scripts from this project that required to set up with Carla environment.

