import os
import json
import cv2
#--------------------------------每张图片对应一个json文件-----------------------------------------------------------
parent_folder = '/home/lbycdy/work/datasets/OCID-VLG/ARID10/table/top/fruits/seq09'
rgb_folder = os.path.join(parent_folder, 'rgb')
json_folder = os.path.join(parent_folder, 'annojson')

# 获取图片和json文件列表
image_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])

# 检查图片和json文件是否一一对应
for image_file in image_files:
    # 构造对应的json文件名
    json_file = image_file.replace('.png', '.json')
    img = cv2.imread(os.path.join(rgb_folder, image_file))
    if json_file not in json_files:
        print(f"Warning: JSON file for {image_file} not found.")
        continue

    # 读取并检查json文件
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)

        except json.JSONDecodeError as e:
            print(f"Error: JSON decoding error in {json_file}: {e}")

    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            top_left = tuple(map(int, points[0]))
            bottom_right = tuple(map(int, points[1]))
            # 绘制矩形框
            # cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
            print(top_left, bottom_right)
            # cv2.rectangle(img, (270, 406), (354,512), (0, 255, 255), 2)
            # cv2.rectangle(img, (int(288*0.64), int(374*0.48)), (int(348*0.64), int(462*0.48)), (0, 255, 255), 2)
            # cv2.rectangle(img, (int(369*0.64), int(650*0.48)), (int(431*0.64), int(733*0.48)), (0, 255, 255), 2)
            # cv2.rectangle(img, (int(203*0.64), int(433*0.48)), (int(314*0.64), int(619*0.48)), (0, 255, 255), 2)
            cv2.rectangle(img, (int(434 ), int(227)), (int(467), int(260 )), (0, 255, 255), 2)

            # 绘制标签
            cv2.putText(img, shape['label'], (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('Labeled Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 输出检查结果
print("Check completed.")



#------------------------------------------所有图片对应一个json文件---------------------------------------------


# # 生成的 JSON 文件路径
# json_file_path = '/home/lbycdy/work/datasets/OCID-VLG/combined.json'  # 替换为生成的JSON文件路径
#
#
# # 读取 JSON 文件
# with open(json_file_path, 'r') as f:
#     data = json.load(f)
#
# # 遍历 JSON 列表中的每个项
# for item in data:
#     image_path = item['imagePath']
#
#     # 读取图片
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Failed to load image at {image_path}")
#         continue
#
#     # 遍历 shapes 以绘制每个标注框
#     for shape in item['shapes']:
#         if shape['shape_type'] == 'rectangle':
#             points = shape['points']
#             label = shape.get('label', 'Unknown')
#
#             # 获取矩形框的左上角和右下角坐标
#             top_left = tuple(map(int, points[0]))
#             bottom_right = tuple(map(int, points[1]))
#
#             # 绘制矩形框
#             cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
#
#             # 在矩形框上方绘制标签
#             cv2.putText(image, label, (top_left[0], top_left[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#     # 显示结果
#     cv2.imshow('Image with Annotations', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#         #
