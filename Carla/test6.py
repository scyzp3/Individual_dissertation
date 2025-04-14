
"""Test for LiDAR and Camera in CARLA vehicle with YOLOv11"""

from ultralytics import YOLO
import carla
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# load YOLO model
model = YOLO('../yolov11.pt').to('cuda')

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

# set up the camera and LiDAR sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')  # set image width
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')  # set field of view

# set camera position
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # adjust position

# build the LiDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')  # set range
lidar_bp.set_attribute('points_per_second', '100000')  # set points per second
lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # adjust position

# define global variables
point_cloud = None

# define camera callback function
def camera_callback(image):
    global frame
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))  # RGBA format
    frame = frame[:, :, :3]  # delete alpha channel
    print("Camera frame received.")

# define LiDAR callback function
def lidar_callback(data):
    global point_cloud
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    point_cloud = points[:, :3]  # liDAR points [x, y, z]
    print("LiDAR points received. Shape:", point_cloud.shape)

try:
    # add the camera sensor to the vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    # add the LiDAR sensor to the vehicle
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_callback)

    # get the LiDAR and camera transforms
    lidar_transform = lidar_sensor.get_transform()
    camera_transform = camera_sensor.get_transform()

    # calculate the transformation matrix from LiDAR to camera
    def get_lidar_to_camera_transform(lidar_transform, camera_transform):
        """
        calculate the transformation matrix from LiDAR to camera
        """
        # get LiDAR and camera locations
        lidar_location = np.array([lidar_transform.location.x, lidar_transform.location.y, lidar_transform.location.z])
        camera_location = np.array([camera_transform.location.x, camera_transform.location.y, camera_transform.location.z])

        # calculate relative translation
        translation = camera_location - lidar_location

        # calculate rotation matrices
        lidar_rotation = R.from_euler('xyz', [lidar_transform.rotation.roll, lidar_transform.rotation.pitch, lidar_transform.rotation.yaw], degrees=True).as_matrix()
        camera_rotation = R.from_euler('xyz', [camera_transform.rotation.roll, camera_transform.rotation.pitch, camera_transform.rotation.yaw], degrees=True).as_matrix()

        # build the transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = camera_rotation @ lidar_rotation.T
        transform_matrix[:3, 3] = translation
        return transform_matrix

    # get the transformation matrix from LiDAR to camera
    lidar_to_camera_matrix = get_lidar_to_camera_transform(lidar_transform, camera_transform)
    print("LiDAR to Camera Transform Matrix:\n", lidar_to_camera_matrix)

    # get the camera intrinsic matrix
    def get_camera_intrinsics(camera_bp):
        """
        get the camera intrinsic matrix
        """
        fov = float(camera_bp.get_attribute("fov"))
        width = int(camera_bp.get_attribute("image_size_x"))
        height = int(camera_bp.get_attribute("image_size_y"))

        fx = width / (2 * np.tan(np.radians(fov / 2)))
        fy = fx
        cx = width / 2
        cy = height / 2

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    # get the camera intrinsic matrix
    camera_intrinsics = get_camera_intrinsics(camera_bp)
    print("Camera Intrinsics Matrix:\n", camera_intrinsics)  # 调试信息

    # project LiDAR points to image plane
    def project_lidar_to_image(points, lidar_to_camera_matrix, camera_intrinsics):
        """
        project LiDAR points to image plane
        """
        # transform LiDAR points to homogeneous coordinates
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

        # transform LiDAR points to camera coordinates
        points_camera = (lidar_to_camera_matrix @ points_homogeneous.T).T[:, :3]

        # project camera points to image plane
        points_image = (camera_intrinsics @ points_camera.T).T
        points_image = points_image[:, :2] / points_image[:, 2].reshape(-1, 1)

        # mask points that are outside the image boundaries
        mask = (points_image[:, 0] >= 0) & (points_image[:, 0] < 800) & \
               (points_image[:, 1] >= 0) & (points_image[:, 1] < 600)
        return points_image[mask]

    # main loop
    while True:
        if frame is not None and point_cloud is not None:
            # build a copy of the frame for processing
            frame_copy = frame.copy()

            # detect objects in the current frame
            results = model(frame_copy)  # 输入当前帧

            # draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # get bounding box coordinates
                    conf = box.conf[0].cpu().numpy()  # get confidence score
                    cls = box.cls[0].cpu().numpy()  # get class index
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame_copy, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # project LiDAR points to image plane
                    points_image = project_lidar_to_image(point_cloud, lidar_to_camera_matrix, camera_intrinsics)

                    # draw LiDAR points on the frame
                    for point in points_image:
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame_copy, (x, y), 2, (0, 255, 0), -1)  # 绘制绿色点

            # show the frame with detections
            cv2.imshow("Camera with LiDAR Points", frame_copy)

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