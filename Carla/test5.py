
"""Test for LiDAR and Camera in CARLA vehicle"""

import carla
import numpy as np
import pygame
from pygame.locals import K_q

# connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# get the current world settings
blueprint_library = world.get_blueprint_library()

# get the vehicle blueprint
vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # 选择第一个车辆蓝图

# get spawn points
spawn_points = world.get_map().get_spawn_points()

# get the vehicle spawn points
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
camera_bp.set_attribute('image_size_x', '800')  # 设置分辨率
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')  # 设置视野

# set camera position
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # adjust position

# set up the LiDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50.0')  # set range
lidar_bp.set_attribute('channels', '32')  # set number of channels
lidar_bp.set_attribute('points_per_second', '100000')  # set points per second

# set LiDAR position
lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # adjust position

# define global variables
frame = None
lidar_data = None

# define camera callback function
def camera_callback(image):
    global frame
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))  # RGBA
    frame = frame[:, :, :3]  # delete alpha channel

# 定义 LiDAR 回调函数
def lidar_callback(point_cloud):
    global lidar_data
    points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_data = points[:, :3]  # [x, y, z]

try:
    # add the camera sensor to the vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    # add the LiDAR sensor to the vehicle
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_callback)

    # original pygame initialization
    pygame.init()

    # set up the display
    camera_screen = pygame.display.set_mode((800, 600))  # camera window
    lidar_screen = pygame.display.set_mode((800, 600))  # LiDAR window
    pygame.display.set_caption("Camera and LiDAR Visualization")

    # main loop
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_q:  # press 'q' to quit
                    running = False

        # show camera image
        if frame is not None:
            camera_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            camera_screen.blit(camera_surface, (0, 0))
            pygame.display.flip()  # update camera window

        # show LiDAR data
        if lidar_data is not None:
            lidar_screen.fill((0, 0, 0))  # clear LiDAR window
            for point in lidar_data:
                x = int(point[0] * 10 + 400)  # scale and center X axis
                y = int(-point[1] * 10 + 300)  # scale and center Y axis
                if 0 <= x < 800 and 0 <= y < 600:
                    pygame.draw.circle(lidar_screen, (255, 255, 255), (x, y), 1)  # draw point
            pygame.display.flip()  # update LiDAR window

        clock.tick(60)  # limit to 60 FPS

finally:
    # destroy sensors and vehicle
    if 'camera_sensor' in globals():
        camera_sensor.destroy()
    if 'lidar_sensor' in globals():
        lidar_sensor.destroy()
    print("Sensors destroyed.")
    pygame.quit()