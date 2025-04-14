
"""Script to filter labels based on KITTI standard"""

import os

# KITTI standard
KITTI_CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Tram', 'Misc', 'Person_sitting', 'DontCare', 'Vehicles', 'TrafficSigns', 'TrafficLight']


def filter_labels(input_label_dir, output_label_dir):
    """
    mask file with the KITTI standard label

    args:
        input_label_dir: original label file directory
        output_label_dir: masked label file directory
    """
    # ensure the output directory exists
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    # loop through all files in the input directory
    for label_file in os.listdir(input_label_dir):
        if not label_file.endswith('.txt'):
            continue

        input_file_path = os.path.join(input_label_dir, label_file)
        output_file_path = os.path.join(output_label_dir, label_file)

        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if line == '':
                    continue

                # get the object name from the line
                obj_name = line.split(' ')[0]

                # if the object name is in the KITTI_CLASSES, write it to the output file
                if obj_name in KITTI_CLASSES:
                    outfile.write(line + '\n')

        print(f'Processed: {label_file}')


input_label_dir = '../dataset/dataset/kitti2/training/label_2'
output_label_dir = '../dataset/dataset/kitti2/training/label_3'

filter_labels(input_label_dir, output_label_dir)