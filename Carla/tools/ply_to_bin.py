"""
convert ply files to bin files
"""

import os
import open3d as o3d
import numpy as np
from tqdm import tqdm


def ply_to_bin(ply_path, bin_path):
    """core function to convert PLY to BIN"""
    try:
        # read PLY file
        pcd = o3d.io.read_point_cloud(ply_path)

        # get points
        points = np.asarray(pcd.points)

        # process intensity
        if pcd.has_intensity():
            intensity = np.asarray(pcd.intensity).reshape(-1, 1)
        elif pcd.has_colors():
            colors = np.asarray(pcd.colors)
            intensity = (0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]).reshape(-1, 1)
        else:
            intensity = np.zeros((points.shape[0], 1))

        # combine points and intensity
        data = np.hstack((points, intensity)).astype(np.float32)

        # save to BIN file
        data.tofile(bin_path)
        return True
    except Exception as e:
        print(f"\nfailed {ply_path}: {str(e)}")
        return False


def batch_convert(input_dir, output_dir):
    """main function to batch convert PLY files to BIN files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get all PLY files in the input directory
    ply_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.ply')]
    print(f"find {len(ply_files)} ply files to be converted")

    # shown progress bar
    success_count = 0
    with tqdm(total=len(ply_files), desc="loading", unit="file") as pbar:
        for filename in ply_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.ply', '.bin'))

            if ply_to_bin(input_path, output_path):
                success_count += 1
            pbar.update(1)

    print(f"success {success_count}/{len(ply_files)} fail {len(ply_files) - success_count}")


if __name__ == "__main__":
    # path to the input PLY files and output BIN files
    input_dir = "./dataset/Town01/Town01/generated/frames"
    output_dir = "./dataset/Town01/velodyne"

    batch_convert(input_dir, output_dir)
