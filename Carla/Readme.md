# Carla

This folder contains CARLA-based data collection, sensor testing, and fusion experiment scripts. The project targets CARLA 0.9.12 and requires the CARLA Python API.

Place the required files from `example` into the CARLA directory before running the scripts.

## Structure

- `demos/`: standalone tests for LiDAR, camera, YOLO, manual control, and sensor integration.
- `pipelines/`: CARLA-connected pipelines for LiDAR visualization, camera projection, YOLO, DeepSORT, realtime processing, and traffic management.
- `tools/`: data conversion utilities and model test helpers.
- `sep/scripts/`: SFA3D and SFA3D+YOLO experiment entry points.

## Notes

Some experiments separate data collection from processing because running CARLA and the main perception pipeline together can be resource-intensive.

For `sep` experiments, place the test dataset in `dataset/` using KITTI format.

Main scripts:

- `sep/scripts/sfa3d_carla_demo.py`: runs the full system with CARLA.
- `sep/scripts/sfa3d_yolo_fusion_demo.py`: runs SFA3D+YOLO fusion on a test dataset.
- `sep/scripts/sfa3d_yolo_fusion_evaluation.py`: runs experiment evaluation.
