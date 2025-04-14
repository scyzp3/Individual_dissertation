
"""Useful functions for filtering"""

def compute_iou(box1, box2):
    """
    计算两个 2D 边界框的 IoU
    box1, box2: [x_min, y_min, x_max, y_max]
    """
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # 计算交集矩形
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)

    # 交集面积
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 各自面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)

    # 计算 IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def get_bounding_box(points):
    """
    计算 n 个 2D 点的外接矩形 (Bounding Box)

    参数:
    - points: np.array, 形状 (n, 2)，包含 n 个 (x, y) 坐标

    返回:
    - [x_min, y_min, x_max, y_max]
    """
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    return [x_min, y_min, x_max, y_max]


def point_to_box_distance(point, box_min, box_max):
    """
    计算点到长方体的最近距离

    参数:
    - point: (3,) 数组，表示点的坐标
    - box_min: (3,) 数组，表示长方体的最小顶点坐标
    - box_max: (3,) 数组，表示长方体的最大顶点坐标

    返回:
    - distance: 点到长方体的最近距离
    - closest_point: 长方体中离点最近的点
    """
    # 将点限制在长方体内
    closest_point = np.clip(point, box_min, box_max)

    # 计算点到最近点的距离
    distance = np.linalg.norm(point - closest_point)

    return distance, closest_point

def closest_distance_to_origin(cuboid_vertices):
    """
    计算长方体中离原点最近的点及其距离

    参数:
    - cuboid_vertices: (8, 3) 数组，表示长方体的 8 个顶点坐标

    返回:
    - min_distance: 最近的顶点到原点的距离
    - closest_point: 最近的点的坐标
    """
    # 计算长方体的最小和最大顶点坐标
    box_min = np.min(cuboid_vertices, axis=0)
    box_max = np.max(cuboid_vertices, axis=0)

    # 计算原点到长方体的最近距离
    origin = np.array([0, 0, 0])
    min_distance, closest_point = point_to_box_distance(origin, box_min, box_max)

    return min_distance

import numpy as np

import numpy as np

def project_camera_to_image(points_cam, P2, image_width, image_height):
    """
    将相机坐标系下的点云投影到图像坐标系，并返回过滤后的 3D 和 2D 点云

    参数:
    - points_cam: (N, 3) 的点云数据，每一行是相机坐标系下的 (X_c, Y_c, Z_c) 坐标
    - P2: 3x4 相机内参矩阵
    - image_width: 图像的宽度
    - image_height: 图像的高度

    返回:
    - points_cam_filtered: (M, 3) 的过滤后的相机坐标系下的点云数据
    - points_img_filtered: (M, 2) 的过滤后的图像坐标系下的点云投影坐标
    """
    # 将点云转换为齐次坐标 (N, 4)
    points_hom = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))

    # 投影到图像平面 (N, 3)
    points_img = np.dot(P2, points_hom.T).T

    # 归一化 (N, 2)
    points_img = points_img[:, :2] / points_img[:, 2:]

    # 过滤掉不在图像范围内的点
    mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < image_width) & \
           (points_img[:, 1] >= 0) & (points_img[:, 1] < image_height)

    # 返回过滤后的 3D 和 2D 点云
    points_cam_filtered = points_cam[mask]
    points_img_filtered = points_img[mask]

    return points_cam_filtered, points_img_filtered


