"""
Test for run SFA3D and yolo with fusion on own dataset and show results in real-time
"""

import argparse
import sys
import os
import time
import warnings
import zipfile
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from examples.sfa.data_process.transformation import lidar_to_camera_point
from examples.sfa.filter import get_bounding_box, compute_iou, closest_distance_to_origin
from examples.sfa.utils.evaluation_utils import convert_det_to_real_values_with_scores

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch深度学习框架
import numpy as np

# 设置项目根目录路径
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)  # 将项目根目录添加到系统路径以支持模块导入[[1]]

# 导入自定义模块
from data_process.demo2 import Demo_KittiDataset  # KITTI数据集处理类
from models.model_utils import create_model  # 模型创建工具函数
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values  # 评估工具
import config.kitti_config as cnf  # KITTI数据集配置参数
from data_process.transformation import lidar_to_camera_box  # 坐标系转换工具
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes, compute_box_3d, \
    project_to_image, draw_box_3d  # 可视化工具
from data_process.kitti_data_utils import Calibration, get_filtered_lidar  # 相机标定数据处理类
from utils.demo_utils2 import parse_demo_configs, do_detect, download_and_unzip, write_credit  # 工具函数

if __name__ == '__main__':
    configs = parse_demo_configs()  # 解析演示配置参数

    # 模型初始化
    model = create_model(configs)  # 根据配置创建模型结构
    yolo_model = YOLO('../../yolov11.pt').to('cuda')
    # tracker = DeepSort(max_age=100,  # 目标丢失后保留帧数
    #                    n_init=0,  # 初始确认期帧数
    #                    nn_budget=100,
    #                    override_track_class=None,
    #                    half=True,
    #                    bgr=True,
    #                    embedder="mobilenet",
    #                    embedder_wts=None,
    #                    polygon=False,
    #                    today=None,
    #                    max_cosine_distance=0.4,
    #                    )
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))  # 加载预训练权重[[9]]
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    # 设备配置
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # 将模型移动到指定设备（CPU/GPU）
    model.eval()  # 设置为评估模式[[9,15]]

    out_cap = None  # 视频输出句柄（当前未启用）
    demo_dataset = Demo_KittiDataset(configs)  # 初始化KITTI演示数据集
    print(len(demo_dataset))
    CLASS_MAPPING = {
        0: "Pedestrian",
        1: "Vehicle",
        2: "Cyclist"
    }
    # Build the K projection matrix:
    # K = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]
    image_w = 720
    image_h = 360
    fov = 90
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0

    with (torch.no_grad()):  # 禁用梯度计算以提升推理速度
        for sample_idx in range(len(demo_dataset)):
            # 加载BEV（鸟瞰图）和前视角图像数据
            metadatas, bev_map, img_rgb = demo_dataset.load_bevmap_front(sample_idx)
            origin_cloud = demo_dataset.get_lidar(sample_idx)

            # 执行目标检测
            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)

            # 可视化处理
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 张量转numpy并归一化
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))  # 调整BEV图像尺寸
            bev_map = draw_predictions(bev_map, detections, configs.num_classes)  # 绘制预测框[[3]]
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)  # 旋转180度以正确显示方向

            # 前视角图像处理
            img_path = metadatas['img_path'][0]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # RGB转BGR格式（OpenCV标准）
            calib = Calibration(configs.calib_path)  # 加载相机标定参数
            # kitti_dets = convert_det_to_real_values(detections)  # 将检测结果转换为真实坐标系[[9,15]]
            kitti_dets, scores, labels = convert_det_to_real_values_with_scores(detections)  # 将检测结果转换为真实坐标系[[9,15]]
            # cloud_3d_4= get_filtered_lidar(origin_cloud, cnf.boundary)
            # cloud_3d_3 = cloud_3d_4[:, :3]
            # cloud_3d = lidar_to_camera_point(cloud_3d_3,calib.V2C,calib.R0)
            # # filtered_3d, filtered_2d = project_camera_to_image(cloud_3d, calib.P2, image_w, image_h)
            # cloud_3d_hom = np.hstack((cloud_3d, np.ones((cloud_3d.shape[0], 1))))
            #
            # # 扩展 K 为 3x4 的矩阵
            # K_hom = np.hstack((K, np.zeros((3, 1))))
            #
            # # 投影到图像平面 (3, N)
            # points_2d_hom = np.dot(K_hom, cloud_3d_hom.T)
            #
            # # 归一化 (N, 2)
            # points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]
            # points_2d = points_2d.T  # 转置为 (N, 2)
            #
            # mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_w) & \
            #        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_h)



            results = yolo_model(img_bgr)

            # 将YOLO检测结果转换为DeepSORT格式 (xywh + confidence + class)
            # detections_for_deepsort = []
            # if results:
            #     for result in results:
            #         for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            #             x1, y1, x2, y2 = map(int, box)
            #             w, h = x2 - x1, y2 - y1
            #             detection = ([x1, y1, w, h], float(conf), int(cls_id))
            #             detections_for_deepsort.append(detection)

            for result in results:
                 boxes = result.boxes  # 检测框对象
                 yolo_visible = np.zeros(len(boxes))
                 output_conf = np.zeros(len(boxes))

            # 使用DeepSORT更新跟踪器（核心跟踪逻辑）
            # tracks = tracker.update_tracks(detections_for_deepsort, frame=img_bgr)

            # for track in tracks:
            #     if not track.is_confirmed() or track.time_since_update > 1:
            #         continue
            #
            #     # 修复1：安全获取track_id
            #     track_id = str(track.track_id)
            #     color_seed = hash(track_id) % 2
            #     color = (0, 255, 0) if color_seed == 0 else (0, 0, 255)
            #
            #     # 修复2：安全获取边界框
            #     ltrb = track.to_ltrb()
            #     x1, y1, x2, y2 = map(int, ltrb)
            #
            #     # 修复3：安全获取类别和置信度
            #     class_id = int(track.det_class) if track.det_class is not None else -1
            #     class_name = yolo_model.names.get(class_id, "Unknown")  # 防御性字典访问
            #     conf = track.det_conf if track.det_conf is not None else 0.0
            #
            #     # 安全生成标签
            #     id_label = f"ID:{track_id} {class_name} {conf:.2f}"
            #
            #     # 绘制边界框和标签
            #     cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(img_bgr, id_label, (x1, max(10, y1 - 10)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 提取当前跟踪框的LiDAR点（使用跟踪后的修正框）
                # box_mask = (u_coord >= x1) & (u_coord <= x2) & (v_coord >= y1) & (v_coord <= y2)
                # box_points = points_2d[box_mask]
                # box_distances = distances[box_mask]

                # 计算并显示平均距离（与原有逻辑兼容）
                # 修改后的LiDAR点距离处理部分
                # if len(box_distances) > 0:
                #     # 计算最近距离（关键修改点）
                #     min_distance = np.min(box_distances)
                #     distance_label = f"Nearest: {min_distance:.2f}m"  # 标签更名
                #
                #     # 添加距离警告颜色（可选功能）
                #     # if min_distance < 5.0:  # 5米内显示红色警告
                #     #     color = (0, 0, 255)
                #     # else:
                #     #     color = (0, 255, 0)
                #
                #     # 显示距离（带动态颜色）
                #     cv2.putText(img_bgr, distance_label, (x1, y2 + 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # else:
                #     cv2.putText(img_bgr, "No Points", (x1, y2 + 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # if len(kitti_dets) > 0:
            #     kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)  # 坐标系转换
            #     img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)  # 绘制检测框[[9,15]]

            sfa_visible = np.zeros(len(kitti_dets))
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)  # 坐标系转换
                for box_idx, label in enumerate(kitti_dets):
                    cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
                    corners_3d = compute_box_3d(dim, location, ry)
                    corners_2d = project_to_image(corners_3d, calib.P2)
                    bb = get_bounding_box(corners_2d)
                    match = 0
                    for result in results:
                        # print(len(results))
                        boxes = result.boxes  # 检测框对象
                        for idx,box in enumerate(boxes):
                            if (compute_iou(bb, map(int, box.xyxy[0]))) > 0.3 :
                                x1, y1, x2, y2 = map(int, bb)
                                yolo_conf = box.conf
                                sfa_conf = scores[box_idx]
                                output_conf[idx] = max(yolo_conf, sfa_conf)
                                yolo_visible[idx] = 1
                                img_bgr = draw_box_3d(img_bgr, corners_2d, color=cnf.colors[int(cls_id)])
                                distance = closest_distance_to_origin(corners_3d)
                                distance_label = f"{distance:.2f}m"
                                cv2.putText(img_bgr, distance_label, (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                match = 1
                                break
                        if scores[box_idx] > 0.6 and match == 0:
                            x1, y1, x2, y2 = map(int, bb)
                            sfa_visible[box_idx] = 1
                            img_bgr = draw_box_3d(img_bgr, corners_2d, color=cnf.colors[int(cls_id)])
                            label = f"{CLASS_MAPPING.get(labels[box_idx])} {scores[box_idx]:.2f}"
                            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            distance = closest_distance_to_origin(corners_3d)
                            distance_label = f"{distance:.2f}m"
                            cv2.putText(img_bgr, distance_label, (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # x1, y1, x2, y2 = map(int, bb)
                        # clp, distance = find_closest_point(cloud_3d, cloud_2d, x1, x2, y1, y2)
                        # if clp is not None:
                        #     distance_label = f"{distance:.2f}m"
                        # else:
                        #     distance_label = "No Points"
                        # cv2.putText(img_bgr, distance_label, (x1, y2 + 20),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for idx, box in enumerate(boxes):
                # 提取边界框坐标、类别 ID 和置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
                class_id = int(box.cls)  # 类别 ID
                conf = float(box.conf)  # 置信度

                # 防御性字典访问类别名称
                class_name = yolo_model.names.get(class_id, "Unknown")

                # 生成标签
                label = f"{class_name} {conf:.2f}"

                # 根据类别生成不同的颜色

                color_seed = hash(class_name) % 2  # 根据类别生成不同的颜色
                color = (0, 255, 0)
                # if color_seed == 0 else (0, 0, 255)

                # 绘制边界框和标签
                if yolo_visible[idx] == 0:
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(img_bgr, label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # clp, distance = find_closest_point(cloud_3d, cloud_2d, x1, x2, y1, y2)
                # if clp is not None:
                #     distance_label = f"{distance:.2f}m"
                # else:
                #     distance_label = "No Points"
                # cv2.putText(img_bgr, distance_label, (x1, y2 + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)  # 合并BEV和RGB图像[[9,15]]
            # if out_cap is None: # 视频输出句柄初始化
            #     out_cap_h, out_cap_w = bev_map.shape[:2]
            #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #     out_path = os.path.join(configs.results_dir, '{}_front.avi'.format(configs.foldername))
            #     print('Create video writer at {}'.format(out_path))
            #     out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))
            #
            # out_cap.write(bev_map)
            # 删除原有的VideoWriter初始化代码
            # 直接显示BEV地图
            cv2.imshow('BEV Map', out_img)
            # 按Q键退出显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_cap:
            out_cap.release()
        cv2.destroyAllWindows()
