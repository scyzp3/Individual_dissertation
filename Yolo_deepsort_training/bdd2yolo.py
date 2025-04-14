import os
import json

# BDD100K 数据集的路径
bdd100k_path = "datasets/bdd100k/val"
output_path = "datasets/bdd100k/val/labels"
# 类别映射，将BDD100K类别映射为YOLO标号
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
    将BDD100K的数据格式转换为YOLO格式
    :param data: BDD100K原始数据
    :return: YOLO格式的标注
    """
    image_width = 1280  # 固定宽度
    image_height = 720  # 固定高度

    yolo_annotations = []
    for obj in data['frames'][0]['objects']:
        category = obj['category']
        if category not in category_map:
            continue  # 如果类别不在映射中，跳过该对象
        class_id = category_map[category]

        # 2D bounding box 转化为 YOLO 格式: (center_x, center_y, width, height)
        x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
        x2, y2 = obj['box2d']['x2'], obj['box2d']['y2']

        # 计算 bounding box 中心点和宽高
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # 保存为YOLO格式：class_id center_x center_y width height
        yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")

    return yolo_annotations



def process_bdd100k_annotations():
    """
    处理BDD100K数据集中的所有标注，转换为YOLO格式并保存
    """
    annotations_path = os.path.join(bdd100k_path, 'labels')  # 进入labels目录
    images_path = os.path.join(bdd100k_path, 'images')  # 进入images目录

    # 遍历BDD100K中的标注文件
    for annotation_file in os.listdir(annotations_path):
        if annotation_file.endswith(".json"):
            # 读取标注文件
            with open(os.path.join(annotations_path, annotation_file), 'r') as f:
                data = json.load(f)

            # 获取图像宽高信息
            image_name = data['name'] + ".jpg"  # 假设图像文件为jpg格式
            image_path = os.path.join(images_path, image_name)
            if not os.path.exists(image_path):
                continue  # 如果图像文件不存在，则跳过

            # 转换为YOLO格式
            yolo_annotations = convert_to_yolo_format(data)

            # 创建YOLO格式的文本文件
            yolo_file_path = os.path.join(output_path, data['name'] + ".txt")
            with open(yolo_file_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

            print(f"Processed {annotation_file} -> {yolo_file_path}")


if __name__ == '__main__':
    process_bdd100k_annotations()
