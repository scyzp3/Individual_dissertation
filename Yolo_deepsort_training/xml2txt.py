"""Convert VOC-style XML annotations to YOLO label files."""

import os
import xml.etree.ElementTree as ET

input_dir = "ISS-Dataset/labels/train"
output_dir = "datasets/train/labels"
image_dir = "datasets/train/images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

classes = []

for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    image_filename = root.find("filename").text
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"warning: {image_path} not exist")
        continue

    yolo_labels = []

    for obj in root.findall("object"):
        # Build the class list from the annotations so IDs stay consistent in this run.
        class_name = obj.find("name").text
        if class_name not in classes:
            classes.append(class_name)
        class_id = classes.index(class_name)

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # YOLO expects normalized center_x, center_y, width, height.
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_labels))

print("Completed, save at:", output_dir)
print("Class:", classes)
