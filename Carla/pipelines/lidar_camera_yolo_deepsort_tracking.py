#!/usr/bin/env python

"""
LiDAR projection on RGB camera example with YOLO detection and DeepSORT tracking.
"""

import glob
import os
import sys
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def sensor_callback(data, queue):
    """
    This simple callback just stores the data on a thread safe Python Queue
    to be retrieved from the "main thread".
    """
    queue.put(data)


def tutorial(args):
    """
    This function is intended to be a tutorial on how to retrieve data in a
    synchronous way, and project 3D points from a lidar to a 2D camera.
    """
    # YOLO provides per-frame detections; DeepSORT links them across frames.
    model = YOLO('../yolov11.pt').to('cuda')
    tracker = DeepSort(max_age=100,
                       n_init=2,
                       nn_budget=100,
                       override_track_class=None,
                       half=True,
                       bgr=True,
                       embedder="mobilenet",
                       embedder_wts=None,
                       polygon=False,
                       today=None,
                       max_cosine_distance=0.4,
                       )

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle = None
    camera = None
    lidar = None

    try:
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]

        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        if args.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        # Keep camera and LiDAR rigidly attached to the ego vehicle.
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[3])
        vehicle.set_autopilot(True)
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # Camera intrinsic matrix for LiDAR-to-image projection.
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        image_queue = Queue()
        lidar_queue = Queue()

        # Queues let the main loop consume synchronized sensor frames after world.tick().
        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame

            try:
                image_data = image_queue.get(True, 2.0)
                lidar_data = lidar_queue.get(True, 2.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            assert image_data.frame == lidar_data.frame == world_frame
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Lidar: %d" %
                (frame, args.frames, world_frame, image_data.frame, lidar_data.frame) + ' ')
            sys.stdout.flush()

            # Convert CARLA raw buffers into OpenCV/Numpy arrays.
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            p_cloud_size = len(lidar_data)
            p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

            intensity = np.array(p_cloud[:, 3])

            local_lidar_points = np.array(p_cloud[:, :3]).T

            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

            lidar_2_world = lidar.get_transform().get_matrix()

            world_points = np.dot(lidar_2_world, local_lidar_points)

            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            sensor_points = np.dot(world_2_camera, world_points)

            # CARLA uses UE coordinates; OpenCV camera space is (x right, y down, z forward).
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

            points_2d = np.dot(K, point_in_camera_coords)

            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            points_2d = points_2d.T
            intensity = intensity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]
            intensity = intensity[points_in_canvas_mask]

            u_coord = points_2d[:, 0].astype(int)
            v_coord = points_2d[:, 1].astype(int)

            intensity = 4 * intensity - 3
            color_map = np.array([
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(int).T


            distances = np.linalg.norm(p_cloud[:, :3], axis=1)
            distances = distances[points_in_canvas_mask]

            im_array_bgr = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

            # Convert YOLO boxes to DeepSORT's xywh input format.
            results = model(im_array_bgr)

            detections_for_deepsort = []
            if results:
                for result in results:
                    for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        detection = ([x1, y1, w, h], float(conf), int(cls_id))
                        detections_for_deepsort.append(detection)

            tracks = tracker.update_tracks(detections_for_deepsort, frame=im_array_bgr)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = str(track.track_id)
                color_seed = hash(track_id) % 2
                color = (0, 255, 0) if color_seed == 0 else (0, 0, 255)

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                class_id = int(track.det_class) if track.det_class is not None else -1
                class_name = model.names.get(class_id, "Unknown")
                conf = track.det_conf if track.det_conf is not None else 0.0

                id_label = f"ID:{track_id} {class_name} {conf:.2f}"

                cv2.rectangle(im_array_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im_array_bgr, id_label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                box_mask = (u_coord >= x1) & (u_coord <= x2) & (v_coord >= y1) & (v_coord <= y2)
                box_points = points_2d[box_mask]
                box_distances = distances[box_mask]

                # Estimate object distance from projected LiDAR points inside the tracked box.
                if len(box_distances) > 0:
                    min_distance = np.min(box_distances)
                    distance_label = f"Nearest: {min_distance:.2f}m"

                    if min_distance < 5.0:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(im_array_bgr, distance_label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(im_array_bgr, "No Points", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for i in range(len(points_2d)):
                # Overlay projected LiDAR points on the camera image.
                u, v = u_coord[i], v_coord[i]
                color = color_map[i][::-1]
                im_array_bgr[v, u] = color

            cv2.imshow("CARLA Camera with Tracking", im_array_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        world.apply_settings(original_settings)

        if camera:
            camera.destroy()
        if lidar:
            lidar.destroy()
        if vehicle:
            vehicle.destroy()

        cv2.destroyAllWindows()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=3000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=2000,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=2,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
