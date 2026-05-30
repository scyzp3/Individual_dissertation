#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiDAR projection to camera image with YOLOv11 detection
"""

import sys
import argparse
import queue
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import cv2
from ultralytics import YOLO  # 使用 YOLOv11
import carla

# YOLOv11 model path
YOLOv11_MODEL_PATH = "../yolov11.pt"

# colormap for LiDAR intensity
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # 归一化到 [0, 1]


def lidar_callback(point_cloud, point_list):
    """process LiDAR data"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # get LiDAR intensity and color
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
    ]

    # get LiDAR points
    points = data[:, :-1]
    points[:, :1] = -points[:, :1]  # 调整 y 轴方向

    # update Open3D point cloud
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def camera_callback(image, image_queue):
    """process camera data"""
    # convert image to BGR format
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 去除 Alpha 通道

    # change color order from RGBA to BGR
    array = np.ascontiguousarray(array)

    # put image into queue
    if image_queue.full():
        image_queue.get()
    image_queue.put(array)


def yolo_detection(image, model):
    """
    use YOLOv11 for object detection
    :param image: input image
    :param model: YOLOv11
    :return: processed image and detection results
    """
    results = model(image)
    detections = []

    # process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            detections.append([x1, y1, x2, y2, confidence, class_id])

            # draw detection box
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, detections


def get_sensor_calibration(lidar, camera):
    """get LiDAR and Camera calibration parameters"""
    # get LiDAR and Camera transforms
    lidar_transform = lidar.get_transform()
    camera_transform = camera.get_transform()

    # compute LiDAR to Camera transformation matrix
    lidar_to_world = lidar_transform.get_matrix()
    world_to_camera = np.linalg.inv(camera_transform.get_matrix())
    lidar_to_camera = np.dot(world_to_camera, lidar_to_world)

    # compute intrinsic parameters
    fov = 90
    width = 800
    height = 600
    fx = width / (2 * np.tan(np.radians(fov / 2)))
    fy = height / (2 * np.tan(np.radians(fov / 2)))
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    distortion = np.zeros(5)

    return lidar_to_camera, K, distortion


def project_lidar_to_camera(points, lidar_to_camera, K, distortion):
    """project LiDAR points to Camera image plane"""
    # convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    #convert points to Camera coordinates
    points_camera = np.dot(lidar_to_camera, points_homogeneous.T).T[:, :3]

    # project points to image plane
    points_image, _ = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3), K, distortion)
    points_image = np.squeeze(points_image)

    return points_image


def draw_lidar_on_image(image, points_image, colors):
    """draw LiDAR points on image"""
    for point, color in zip(points_image, colors):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 1, color * 255, -1)


def fuse_lidar_yolo(image, detections, points_image, colors):
    """fuse LiDAR and YOLO data"""
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection

        # get points in the detection box
        mask = (points_image[:, 0] >= x1) & (points_image[:, 0] <= x2) & \
               (points_image[:, 1] >= y1) & (points_image[:, 1] <= y2)
        points_in_box = points_image[mask]
        colors_in_box = colors[mask]

        # draw detection box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_lidar_on_image(image, points_in_box, colors_in_box)

        # compute depth and label
        if len(points_in_box) > 0:
            depth = np.mean(points_in_box[:, 2])
            label = f"{model.names[class_id]} {confidence:.2f} Depth: {depth:.2f}m"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """generate LiDAR sensor blueprint"""
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp


def generate_camera_bp(blueprint_library, delta):
    """generate Camera sensor blueprint"""
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', str(delta))
    return camera_bp


def add_open3d_axis(vis):
    """add Open3D axis to visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]
    ]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    vis.add_geometry(axis)


def main(arg):
    # connect to CARLA server
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(5.0)
    world = client.get_world()

    try:
        # save original settings
        original_settings = world.get_settings()
        settings = world.get_settings()

        # set synchronous mode
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05
        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        # generate vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        vehicle_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(arg.no_autopilot)

        # generate LiDAR sensor
        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # generate Camera sensor
        camera_bp = generate_camera_bp(blueprint_library, delta)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # organize point cloud data
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: lidar_callback(data, point_list))

        # organize camera data
        image_queue = queue.Queue(maxsize=10)  # 限制队列大小
        camera.listen(lambda image: camera_callback(image, image_queue))

        # load YOLOv11 model
        model = YOLO(YOLOv11_MODEL_PATH)

        # organize Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar with YOLOv11',
            width=960,
            height=540,
            left=480,
            top=270
        )
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        if arg.show_axis:
            add_open3d_axis(vis)

        # get LiDAR and Camera calibration
        lidar_to_camera, K, distortion = get_sensor_calibration(lidar, camera)

        # main loop
        frame = 0
        dt0 = datetime.now()
        while True:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)

            vis.poll_events()
            vis.update_renderer()

            # tick world
            world.tick()

            # process camera image
            if not image_queue.empty():
                image = image_queue.get()

                # detect objects using YOLOv11
                processed_image, detections = yolo_detection(image, model)

                # get LiDAR points
                points = np.asarray(point_list.points)
                points_image = project_lidar_to_camera(points, lidar_to_camera, K, distortion)

                # fuse LiDAR and YOLO data
                fuse_lidar_yolo(processed_image, detections, points_image, np.asarray(point_list.colors))

                # compute depth
                cv2.imshow('Camera Output with YOLOv11 and LiDAR', processed_image)
                cv2.waitKey(1)

            # compute FPS
            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

    finally:
        # restore original settings
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        # destroy actors
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'lidar' in locals():
            lidar.destroy()
        if 'camera' in locals():
            camera.destroy()
        if 'vis' in locals():
            vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host', metavar='H', default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)'
    )
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port of CARLA Simulator (default: 2000)'
    )
    argparser.add_argument(
        '--no-rendering', action='store_true',
        help='use the no-rendering mode which will provide some extra performance'
    )
    argparser.add_argument(
        '--no-autopilot', action='store_false',
        help='disables the autopilot so the vehicle will remain stopped'
    )
    argparser.add_argument(
        '--show-axis', action='store_true',
        help='show the cartesian coordinates axis'
    )
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='model3',
        help='actor filter (default: "vehicle.*")'
    )
    argparser.add_argument(
        '--upper-fov', default=15.0, type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)'
    )
    argparser.add_argument(
        '--lower-fov', default=-25.0, type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)'
    )
    argparser.add_argument(
        '--channels', default=64.0, type=float,
        help='lidar\'s channel count (default: 64)'
    )
    argparser.add_argument(
        '--range', default=100.0, type=float,
        help='lidar\'s maximum range in meters (default: 100.0)'
    )
    argparser.add_argument(
        '--points-per-second', default=500000, type=int,
        help='lidar\'s points per second (default: 500000)'
    )
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')