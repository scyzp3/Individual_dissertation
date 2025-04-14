"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
"""

import sys
import os
from builtins import int
from glob import glob

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_data_utils import get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap
import config.kitti_config as cnf


class Demo_KittiDataset(Dataset):
    def __init__(self, configs):
        self.dataset_dir = os.path.join(configs.dataset_dir)
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        self.image_dir = os.path.join(self.dataset_dir, "image")
        self.lidar_dir = os.path.join(self.dataset_dir, "velodyne")
        self.label_dir = os.path.join(self.dataset_dir, "label_2")
        self.sample_id_list = sorted(glob(os.path.join(self.lidar_dir, '*.bin')))
        self.sample_id_list = [float(os.path.basename(fn)[:-4]) for fn in self.sample_id_list]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        pass

    def load_bevmap_front(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, img_rgb

    def load_bevmap_front_vs_back(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)

        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        back_lidar = get_filtered_lidar(lidarData, cnf.boundary_back)
        back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
        back_bevmap = torch.from_numpy(back_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, back_bevmap, img_rgb

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_labels(self, idx, target_classes=None):
        """
        解析KITTI格式的标签文件，提取2D检测框及类别信息
        :param label_path: 标签文件路径，如"000001.txt"
        :param target_classes: 需要保留的目标类别列表，默认保留全部
        :return: 结构化数据 (n,5) array格式：[xmin, ymin, xmax, ymax, class_id]
        """
        # KITTI类别到ID映射（可根据需要扩展）
        class_mapping = {
            'Car': 1,
            'Pedestrian': 0,
            'Cyclist': 2,
            'Vehicles': 1,
            'Van': 1,
            'Truck': 1,
            'Person_sitting': 0,
            # 'TrafficSigns': 4,
            # 'TrafficLight': 4
        }

        valid_objects = []

        label_path = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))

        with open(label_path, 'r') as f:
            for line in f:
                components = line.strip().split()

                # 验证数据完整性（每行至少15个字段）
                if len(components) < 15:
                    continue

                class_name = components[0]

                # 过滤不需要的类别
                if target_classes and class_name not in target_classes:
                    continue

                # 跳过被截断超过50%或高度小于25像素的目标
                # truncation = float(components[1])
                # bbox_height = float(components[7]) - float(components[5])
                # if truncation > 0.5 or bbox_height < 10:
                #     continue

                # 提取2D边界框坐标（转换为0-based索引）
                xmin = max(0, float(components[4]))
                ymin = max(0, float(components[5]))
                xmax = float(components[6])
                ymax = float(components[7])

                # 处理无效坐标
                if xmin >= xmax or ymin >= ymax:
                    continue

                # 类别映射与统一
                class_id = class_mapping.get(class_name, -1)
                if class_id == -1:
                    continue  # 跳过未定义类别

                valid_objects.append([xmin, ymin, xmax, ymax, class_id])

        if not valid_objects:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(valid_objects, dtype=np.float32)


def compute_map_recall(pred_boxes, gt_boxes, iou_threshold=0.3):
    """
    计算 mAP 和 Recall（不考虑类别）
    :param pred_boxes: 预测框 (m,4) [xmin, ymin, xmax, ymax]
    :param gt_boxes: 真实框 (n,4) [xmin, ymin, xmax, ymax]
    :param iou_threshold: IoU 阈值（默认 0.5）
    :return: (mAP, recall)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, 0.0

    tp = 0  # True Positives（IoU > threshold 的预测框）
    fp = 0  # False Positives（IoU <= threshold 或冗余检测）
    fn = 0  # False Negatives（未被检测到的真实框）

    used_gt_indices = set()

    for pred_box in pred_boxes:
        max_iou = 0.0
        best_gt_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if i in used_gt_indices:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = i

        if max_iou >= iou_threshold:
            tp += 1
            used_gt_indices.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall

def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU（Intersection over Union）
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: IoU 值（0~1）
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集区域
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou
