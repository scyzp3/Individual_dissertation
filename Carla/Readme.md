# Carla
To run the code in this folder, you need to have the CARLA simulator(Version 0.9.12) installed. Please place every files in the folder "example" in the CARLA directory.

The scripts in this folder are designed to work with the CARLA simulator. Top-level scripts are grouped by purpose:

- demos/: standalone tests for LiDAR, camera, YOLO, manual control, and sensor integration.
- pipelines/: CARLA-connected pipelines for LiDAR visualization, camera projection, YOLO, DeepSORT, realtime processing, and traffic management.
- tools/: conversion utilities and screen-region model tests.
- sep/scripts/: runnable SFA3D and SFA3D+YOLO experiment scripts.

However, the computer spends too much performance on the simulator, which leads to poor performance on the main system, so we separate the data collection and processing into two parts. Other files are useful tools used in this project.

In the sep folder, we have the latest version of the code that can run separately with the CARLA server. To run that, please put the test dataset in the dataset directory and save it in KITTI format. The directories here contain library files and configuration files for SFA3D. sep/scripts/sfa3d_carla_demo.py runs the whole system together with the CARLA simulator. sep/scripts/sfa3d_yolo_fusion_demo.py runs the process on the test dataset. sep/scripts/sfa3d_yolo_fusion_evaluation.py is used for experiments and evaluation. Other files are useful tools used in this project.
