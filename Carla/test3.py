
"""Test for YOLOv11 on vehicle camera in CARLA"""

from ultralytics import YOLO
import carla
import cv2
import numpy as np

# load YOLO model
model = YOLO('../yolov11.pt').to('cuda')

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

# build the camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')  # 设置分辨率
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')  # 设置视野

# set camera position
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 调整位置

# define global variables
frame = None

# define camera callback function
def camera_callback(image):
    global frame
    frame = np.frombuffer(image.raw_data, dtype=np.uint8)
    frame = frame.reshape((image.height, image.width, 4))  # RGBA
    frame = frame[:, :, :3]  # delete alpha channel

try:
    # add the camera sensor to the vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    # main loop
    while True:
        if frame is not None:
            # add a copy of the frame for processing
            frame_copy = frame.copy()

            # detect objects in the current frame
            results = model(frame_copy)

            # draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # get bounding box coordinates
                    conf = box.conf[0].cpu().numpy()  # get confidence score
                    cls = box.cls[0].cpu().numpy()  #get class index
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame_copy, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # show the frame with detections
            cv2.imshow("Camera", frame_copy)

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # destroy the camera sensor
    if 'camera_sensor' in globals():
        camera_sensor.destroy()
    print("Sensors destroyed.")
    cv2.destroyAllWindows()