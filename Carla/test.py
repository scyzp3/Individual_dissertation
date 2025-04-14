
"""Test for Lidar and Camera in CARLA"""

import carla
import cv2
import numpy as np
import matplotlib.pyplot as plt

# connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# get the current world settings
blueprint_library = world.get_blueprint_library()

# get the vehicle blueprint
vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # select the first vehicle blueprint

# get spawn points
spawn_points = world.get_map().get_spawn_points()

# try to spawn the vehicle at a random spawn point
vehicle = None
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        print(f"Vehicle spawned successfully at {spawn_point.location}.")
        break
else:
    print("Failed to spawn vehicle at any spawn point.")
    exit()

# build the camera and LiDAR sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')  # set image width
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')  # set field of view

# set camera position
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # adjust position

# build the LiDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50.0')  # set range
lidar_bp.set_attribute('channels', '32')  # set number of channels
lidar_bp.set_attribute('points_per_second', '100000')  # set points per second

# set LiDAR position
lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 调整位置

# set up the camera and LiDAR sensors
frame = None
lidar_data = None

# set up the camera and LiDAR sensors
def camera_callback(image):
    global frame
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))  # RGBA format
    frame = frame[:, :, :3]  # delete alpha channel

# set up the LiDAR callback
def lidar_callback(point_cloud):
    global lidar_data
    lidar_data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    lidar_data = lidar_data.reshape(-1, 4)  #  [x, y, z, intensity]
    print(f"LiDAR data shape: {lidar_data.shape}")  # print shape
    print(f"LiDAR data sample: {lidar_data[:5]}")  # print first 5 points

try:
    # add camera sensor to vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    # add LiDAR sensor to vehicle
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_callback)

    # build the display window
    plt.ion()  # open interactive mode
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=1)  #originally ax.scatter([], [], s=1, c='b', marker='o')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('LiDAR Point Cloud')

    # main loop
    while True:
        if frame is not None:
            # show camera image
            cv2.imshow("Camera", frame)

        if lidar_data is not None:
            # update LiDAR point cloud
            sc.set_offsets(lidar_data[:, :2])  # 只显示 X 和 Y 坐标
            ax.set_xlim(lidar_data[:, 0].min(), lidar_data[:, 0].max())  # 动态设置 X 轴范围
            ax.set_ylim(lidar_data[:, 1].min(), lidar_data[:, 1].max())  # 动态设置 Y 轴范围
            fig.canvas.draw()  # 重新绘制图形
            fig.canvas.flush_events()  # 刷新事件循环

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # destroy sensors and vehicle
    if 'camera_sensor' in globals():
        camera_sensor.destroy()
    if 'lidar_sensor' in globals():
        lidar_sensor.destroy()
    print("Sensors destroyed.")
    cv2.destroyAllWindows()
    plt.close()