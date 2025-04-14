
"""Script to convert XML annotations to YOLO format."""

import os
import xml.etree.ElementTree as ET

# path
input_dir = "ISS-Dataset/labels/train"
output_dir = "datasets/train/labels"
image_dir = "datasets/train/images"

# output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# dynamic class
classes = []

# loop through all XML files in the input directory
for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    # read XML file
    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    # get image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # validate image file
    image_filename = root.find("filename").text
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"warning: {image_path} not exist")
        continue

    # original image
    yolo_labels = []

    # loop through all objects in the XML file
    for obj in root.findall("object"):
        # get class name
        class_name = obj.find("name").text
        if class_name not in classes:
            classes.append(class_name)  # 动态添加新类别
        class_id = classes.index(class_name)

        # get bounding box coordinates
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        # save YOLO format
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # output YOLO labels
    txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_labels))

# 输出类别信息
print("Completed, save at:", output_dir)
print("Class:", classes)
