
"""
Test for run SFA3D on KITTI dataset and save the results
"""

import argparse
import sys
import os
import time
import warnings
import zipfile
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
from data_process.demo_dataset import Demo_KittiDataset  # KITTI数据集处理类
from models.model_utils import create_model  # 模型创建工具函数
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values  # 评估工具
import config.kitti_config as cnf  # KITTI数据集配置参数
from data_process.transformation import lidar_to_camera_box  # 坐标系转换工具
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes  # 可视化工具
from data_process.kitti_data_utils import Calibration  # 相机标定数据处理类
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit  # 工具函数

if __name__ == '__main__':
    configs = parse_demo_configs()  # 解析演示配置参数

    # 模型初始化
    model = create_model(configs)  # 根据配置创建模型结构
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

    with torch.no_grad():  # 禁用梯度计算以提升推理速度
        for sample_idx in range(len(demo_dataset)):
            # 加载BEV（鸟瞰图）和前视角图像数据
            metadatas, bev_map, img_rgb = demo_dataset.load_bevmap_front(sample_idx)

            # 执行目标检测
            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)

            # 可视化处理
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 张量转numpy并归一化
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))  # 调整BEV图像尺寸
            bev_map = draw_predictions(bev_map, detections, configs.num_classes)  # 绘制预测框[[3]]
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)  # 旋转180度以正确显示方向

            # 直接显示BEV地图
            cv2.imshow('BEV Map', bev_map)
            # 按Q键退出显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_cap:
            out_cap.release()
        cv2.destroyAllWindows()
