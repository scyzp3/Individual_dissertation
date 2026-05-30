#!/usr/bin/env python

"""Open3D Lidar and Camera visualization example for CARLA"""

import glob
import os
import sys
import argparse
import time
import queue
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 定义颜色映射
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),     # Building
    (100, 40, 40),    # Fences
    (55, 90, 80),     # Other
    (220, 20, 60),    # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),   # RoadLines
    (128, 64, 128),   # Road
    (244, 35, 232),   # Sidewalk
    (107, 142, 35),   # Vegetation
    (0, 0, 142),      # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),    # TrafficSign
    (70, 130, 180),   # Sky
    (81, 0, 81),      # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),   # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),   # Dynamic
    (45, 60, 150),    # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # 归一化到 [0, 1]


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


def semantic_lidar_callback(point_cloud, point_list):
    """处理语义 LiDAR 数据并更新点云"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
    ]))

    # 提取点云数据并调整坐标系
    points = np.array([data['x'], -data['y'], data['z']]).T

    # 根据语义标签映射颜色
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # 更新 Open3D 点云
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def camera_callback(image, image_queue):
    """处理摄像头数据并保存到队列中"""
    image.save_to_disk('output/%08d.png' % image.frame)
    image_queue.put(image)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """生成 LiDAR 传感器蓝图"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')

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
        if arg.semantic:
            lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
        else:
            lidar.listen(lambda data: lidar_callback(data, point_list))

        # 初始化摄像头图像队列
        image_queue = queue.Queue()
        camera.listen(lambda image: camera_callback(image, image_queue))

        # 初始化 Open3D 可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
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
                # 这里可以添加对图像的处理逻辑

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
        '--semantic', action='store_true',
        help='use the semantic lidar instead, which provides ground truth information'
    )
    argparser.add_argument(
        '--no-noise', action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar'
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
    argparser.add_argument(
        '-x', default=0.0, type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)'
    )
    argparser.add_argument(
        '-y', default=0.0, type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)'
    )
    argparser.add_argument(
        '-z', default=0.0, type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)'
    )
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')