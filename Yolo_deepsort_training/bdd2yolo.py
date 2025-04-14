
"""Convert BDD100K annotations to YOLO format"""

import os
import json

# BDD100K dataset path
bdd100k_path = "datasets/bdd100k/val"
output_path = "datasets/bdd100k/val/labels"
# class mapping for YOLO
category_map = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "bicycle": 4,
    "motorcycle": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "sky": 9,
    "building": 10,
    "road": 11,
    "sidewalk": 12,
    "fence": 13,
    "lane marker": 14,
    "pedestrian": 15,
    "animal": 16,
    "obstacle": 17
}


def convert_to_yolo_format(data):
    """
    convert BDD100K annotations to YOLO format
    """
    image_width = 1280
    image_height = 720

    yolo_annotations = []
    for obj in data['frames'][0]['objects']:
        category = obj['category']
        if category not in category_map:
            continue  # if category not in the mapping, skip it
        class_id = category_map[category]

        # 2D bounding box to YOLO: (center_x, center_y, width, height)
        x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
        x2, y2 = obj['box2d']['x2'], obj['box2d']['y2']

        # compute center_x, center_y, width, height
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # save in YOLO format
        yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")

    return yolo_annotations



def process_bdd100k_annotations():
    """
    process BDD100K annotations and convert to YOLO format
    """
    annotations_path = os.path.join(bdd100k_path, 'labels')
    images_path = os.path.join(bdd100k_path, 'images')

    # load annotations
    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith(".json"):
            # load annotation file
            with open(os.path.join(annotations_path, annotation_file), 'r') as f:
                data = json.load(f)

            # get image name
            image_name = data['name'] + ".jpg"
            image_path = os.path.join(images_path, image_name)
            if not os.path.exists(image_path):
                continue  # if image does not exist, skip it

            # convert to YOLO format
            yolo_annotations = convert_to_yolo_format(data)

            # set output path
            yolo_file_path = os.path.join(output_path, data['name'] + ".txt")
            with open(yolo_file_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

            print(f"Processed {annotation_file} -> {yolo_file_path}")


if __name__ == '__main__':
    process_bdd100k_annotations()
