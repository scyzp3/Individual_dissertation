#!/usr/bin/env python

"""Test for LiDAR and Camera in CARLA vehicle with YOLOv11"""

import sys
import argparse
import queue
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import cv2
from ultralytics import YOLO
import carla

# YOLOv11 path
YOLOv11_MODEL_PATH = "../yolov11.pt"

# define color map for LiDAR intensity
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None00000000000000000000000000000000001
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
]) / 255.0


def lidar_callback(point_cloud, point_list):
    """处理 LiDAR 数据并更新点云"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # 提取强度信息并映射为颜色
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
    ]

    # 提取点云数据并调整坐标系
    points = data[:, :-1]
    points[:, :1] = -points[:, :1]  # 调整 y 轴方向

    # 更新 Open3D 点云
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def camera_callback(image, image_queue):
    """处理摄像头数据并保存到队列中"""
    # 将图像转换为 OpenCV 格式
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 去除 Alpha 通道

    # 复制到一个可写的数组中
    array = np.ascontiguousarray(array)

    # 将图像放入队列
    if image_queue.full():
        image_queue.get()  # 如果队列已满，丢弃最旧的图像
    image_queue.put(array)


def yolo_detection(image, model):
    """
    使用 YOLOv11 进行目标检测
    :param image: 输入的图像 (BGR 格式)
    :param model: YOLOv11 模型
    :return: 带有检测框的图像
    """
    results = model(image)  # 进行目标检测
    detections = results[0].boxes.data.cpu().numpy()  # 获取检测框数据

    # 解析检测结果
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{model.names[int(class_id)]} {confidence:.2f}"

        # 画框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """生成 LiDAR 传感器蓝图"""
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp


def generate_camera_bp(blueprint_library, delta):
    """生成摄像头传感器蓝图"""
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', str(delta))
    return camera_bp


def add_open3d_axis(vis):
    """在 Open3D 可视化中添加坐标系"""
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
    """主函数"""
    # 连接到 CARLA 服务器
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(5.0)
    world = client.get_world()

    try:
        # 保存原始设置
        original_settings = world.get_settings()
        settings = world.get_settings()

        # 设置同步模式
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05
        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        # 生成车辆
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        vehicle_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(arg.no_autopilot)

        # 生成 LiDAR 传感器
        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # 生成摄像头传感器
        camera_bp = generate_camera_bp(blueprint_library, delta)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 初始化 Open3D 点云
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: lidar_callback(data, point_list))

        # 初始化摄像头图像队列
        image_queue = queue.Queue(maxsize=10)  # 限制队列大小
        camera.listen(lambda image: camera_callback(image, image_queue))

        # 加载 YOLOv11 模型
        model = YOLO(YOLOv11_MODEL_PATH)

        # 初始化 Open3D 可视化窗口
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

        # 主循环
        frame = 0
        dt0 = datetime.now()
        while True:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)

            vis.poll_events()
            vis.update_renderer()

            # 推进仿真时间
            world.tick()

            # 处理摄像头图像
            if not image_queue.empty():
                image = image_queue.get()
                # 使用 YOLOv11 进行目标检测
                processed_image = yolo_detection(image, model)
                cv2.imshow('Camera Output with YOLOv11', processed_image)
                cv2.waitKey(1)

            # 计算并显示 FPS
            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

    finally:
        # 恢复原始设置
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        # 销毁传感器和车辆
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'lidar' in locals():
            lidar.destroy()
        if 'camera' in locals():
            camera.destroy()
        if 'vis' in locals():
            vis.destroy_window()
        cv2.destroyAllWindows()  # 关闭 OpenCV 窗口


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