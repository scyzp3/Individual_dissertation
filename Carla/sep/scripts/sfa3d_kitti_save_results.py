"""
Run SFA3D on a KITTI-format dataset and show the BEV results.
"""

import argparse
import sys
import os
import time
import warnings
import zipfile

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit


if __name__ == '__main__':
    configs = parse_demo_configs()

    # Initialize SFA3D.
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)
    print(len(demo_dataset))

    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            # Load one KITTI sample, run SFA3D on its BEV map, and display predictions.
            metadatas, bev_map, img_rgb = demo_dataset.load_bevmap_front(sample_idx)

            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections, configs.num_classes)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            cv2.imshow('BEV Map', bev_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out_cap:
            out_cap.release()
        cv2.destroyAllWindows()
