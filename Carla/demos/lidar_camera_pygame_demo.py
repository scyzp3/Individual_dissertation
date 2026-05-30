
"""Test for LiDAR and Camera in CARLA vehicle"""

import carla
import numpy as np
import pygame
from pygame.locals import K_q

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter('vehicle.*')[0]

spawn_points = world.get_map().get_spawn_points()

vehicle = None
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        print(f"Vehicle spawned successfully at {spawn_point.location}.")
        break
else:
    print("Failed to spawn vehicle at any spawn point.")
    exit()

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50.0')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('points_per_second', '100000')

lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

frame = None
lidar_data = None

def camera_callback(image):
    # CARLA camera frames arrive as BGRA; keep only the RGB channels for display.
    global frame
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))
    frame = frame[:, :, :3]

def lidar_callback(point_cloud):
    # Each LiDAR point is x, y, z, intensity; this demo only plots 2D position.
    global lidar_data
    points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_data = points[:, :3]

try:
    # Attach both sensors to the same vehicle so their views share a frame of reference.
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_callback)

    pygame.init()

    camera_screen = pygame.display.set_mode((800, 600))
    lidar_screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Camera and LiDAR Visualization")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_q:
                    running = False

        if frame is not None:
            camera_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            camera_screen.blit(camera_surface, (0, 0))
            pygame.display.flip()

        if lidar_data is not None:
            lidar_screen.fill((0, 0, 0))
            for point in lidar_data:
                # Scale CARLA meters into window pixels and center the ego vehicle.
                x = int(point[0] * 10 + 400)
                y = int(-point[1] * 10 + 300)
                if 0 <= x < 800 and 0 <= y < 600:
                    pygame.draw.circle(lidar_screen, (255, 255, 255), (x, y), 1)
            pygame.display.flip()

        clock.tick(60)

finally:
    if 'camera_sensor' in globals():
        camera_sensor.destroy()
    if 'lidar_sensor' in globals():
        lidar_sensor.destroy()
    print("Sensors destroyed.")
    pygame.quit()
