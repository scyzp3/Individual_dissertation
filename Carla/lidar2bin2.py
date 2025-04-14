#!/usr/bin/env python

"""convert carla lidar data to KITTI format and save to disk"""

import argparse
import glob
import os
import queue
import random
import sys
import numpy as np
import carla

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# KITTI architecture
KITTI_CONFIG = {
    "lidar": {
        "range": 50.0,
        "channels": 64,
        "points_per_second": 1200000,
        "rotation_frequency": 10.0,
        "upper_fov": 2.0,
        "lower_fov": -24.8
    }
}


def create_save_dirs(save_root):
    """generate save directories"""
    os.makedirs(os.path.join(save_root, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "calib"), exist_ok=True)
    print(f"save in: {save_root}")


def lidar_callback(point_cloud, data_queue):
    """process LiDAR data"""

    raw_data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    data = raw_data.copy().reshape((-1, 4))  # [[关键修复点]]
    # convert to KITTI format
    data[:, 1] = -data[:, 1]  # convert y to -y

    # validate data
    if data.size == 0:
        print("No data received from LiDAR")
        return

    # put data into queue
    data_queue.put((point_cloud.frame, data))


def save_kitti_data(data_queue, save_root):
    """save LiDAR data in KITTI format"""
    while not data_queue.empty():
        frame_id, points = data_queue.get()

        # save point cloud data
        bin_path = os.path.join(save_root, "velodyne", f"{frame_id:06d}.bin")
        points.astype(np.float32).tofile(bin_path)

        # generate calibration file
        calib_content = f"""P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00
P1: 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00
P2: 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
Tr_velo_to_cam: 4.276802e-04 -9.999672e-01 -8.084491e-03 -1.198957e-02
                -7.210626e-03 8.081198e-03 -9.999413e-01 -5.403984e-02
                9.999739e-01 4.859810e-04 -7.206933e-03 -2.921968e-01"""

        calib_path = os.path.join(save_root, "calib", f"{frame_id:06d}.txt")
        with open(calib_path, "w") as f:
            f.write(calib_content)

        print(f"save {frame_id:06d}: cloud {points.shape[0]} point")


def setup_lidar_sensor(world, vehicle, data_queue):
    """setup LiDAR sensor"""
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

    # set attributes
    lidar_bp.set_attribute('range', str(KITTI_CONFIG["lidar"]["range"]))
    lidar_bp.set_attribute('rotation_frequency', str(KITTI_CONFIG["lidar"]["rotation_frequency"]))
    lidar_bp.set_attribute('points_per_second', str(KITTI_CONFIG["lidar"]["points_per_second"]))
    lidar_bp.set_attribute('channels', str(KITTI_CONFIG["lidar"]["channels"]))
    lidar_bp.set_attribute('upper_fov', str(KITTI_CONFIG["lidar"]["upper_fov"]))
    lidar_bp.set_attribute('lower_fov', str(KITTI_CONFIG["lidar"]["lower_fov"]))

    # set transform
    transform = carla.Transform(carla.Location(z=2.4))
    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)

    # register callback
    lidar.listen(lambda data: lidar_callback(data, data_queue))
    return lidar


def main():
    parser = argparse.ArgumentParser(description='KITTI格式LiDAR数据采集')
    parser.add_argument('--host', default='localhost', help='CARLA服务器地址')
    parser.add_argument('--port', type=int, default=3000, help='CARLA服务器端口')
    parser.add_argument('--save-dir',default='./kitti_lidar', help='数据保存路径')
    args = parser.parse_args()

    # originally use 'carla.LidarData'
    data_queue = queue.Queue()
    create_save_dirs(args.save_dir)

    try:
        # connect to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.load_world('Town01')

        # set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # generate random spawn point
        vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # set up LiDAR sensor
        lidar = setup_lidar_sensor(world, vehicle, data_queue)

        # main
        for _ in range(100):  # collect 100 frames
            world.tick()
            save_kitti_data(data_queue, args.save_dir)

    finally:
        # if 'lidar' in locals():
        #     lidar.destroy()
        # if 'vehicle' in locals():
        #     vehicle.destroy()
        # world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        print("completed")


if __name__ == '__main__':
    main()
