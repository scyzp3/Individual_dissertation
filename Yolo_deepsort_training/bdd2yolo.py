"""Convert BDD100K annotations to YOLO format."""

import os
import json

bdd100k_path = "datasets/bdd100k/val"
output_path = "datasets/bdd100k/val/labels"

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
    """Convert one BDD100K frame annotation to YOLO label rows."""
    image_width = 1280
    image_height = 720

    yolo_annotations = []
    for obj in data['frames'][0]['objects']:
        category = obj['category']
        if category not in category_map:
            continue
        class_id = category_map[category]

        x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
        x2, y2 = obj['box2d']['x2'], obj['box2d']['y2']

        # YOLO labels use normalized center_x, center_y, width, height.
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")

    return yolo_annotations


def process_bdd100k_annotations():
    """Process all validation annotations that have a matching image file."""
    annotations_path = os.path.join(bdd100k_path, 'labels')
    images_path = os.path.join(bdd100k_path, 'images')

    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith(".json"):
            with open(os.path.join(annotations_path, annotation_file), 'r') as f:
                data = json.load(f)

            image_name = data['name'] + ".jpg"
            image_path = os.path.join(images_path, image_name)
            if not os.path.exists(image_path):
                continue

            yolo_annotations = convert_to_yolo_format(data)

            yolo_file_path = os.path.join(output_path, data['name'] + ".txt")
            with open(yolo_file_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

            print(f"Processed {annotation_file} -> {yolo_file_path}")


if __name__ == '__main__':
    process_bdd100k_annotations()
