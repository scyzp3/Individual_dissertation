import os
import xml.etree.ElementTree as ET

# 输入和输出路径
# input_dir = "ISS-Dataset/labels/images"  # 替换为 VOC XML 文件夹路径
# output_dir = "ISS-Dataset/labels_txt/images"  # 替换为 YOLO 标签保存路径
# image_dir = "ISS-Dataset/images/images"  # 图片文件夹路径，用于验证 XML 对应的图片是否存在
input_dir = "ISS-Dataset/labels/train"  # 替换为 VOC XML 文件夹路径
output_dir = "datasets/train/labels"  # 替换为 YOLO 标签保存路径
image_dir = "datasets/train/images"  # 图片文件夹路径，用于验证 XML 对应的图片是否存在

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 动态类别列表
classes = []

# 遍历所有 XML 文件
for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    # 解析 XML 文件
    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # 验证图片是否存在
    image_filename = root.find("filename").text
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"警告: 图片文件 {image_path} 不存在，跳过 XML 文件 {xml_file}")
        continue

    # 初始化 YOLO 标签内容
    yolo_labels = []

    # 遍历所有目标
    for obj in root.findall("object"):
        # 获取类别名称
        class_name = obj.find("name").text
        if class_name not in classes:
            classes.append(class_name)  # 动态添加新类别
        class_id = classes.index(class_name)

        # 获取边界框
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # 转换为 YOLO 格式
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        # 保存标签
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # 输出标签文件
    txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_labels))

# 输出类别信息
print("转换完成！YOLO 标签已保存到:", output_dir)
print("检测到的类别:", classes)
