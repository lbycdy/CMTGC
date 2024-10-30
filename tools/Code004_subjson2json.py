import os
import json

# 定义主文件夹路径
main_folder = '/home/lbycdy/work/datasets/OCID-VLG'  # 替换为你的主文件夹路径
output_json_path = '/home/lbycdy/work/datasets/OCID-VLG/combined.json'  # 替换为输出的统一JSON文件路径

# 存储所有json内容的列表
combined_data = []

# 递归遍历主文件夹下的所有子文件夹
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'imageData' in data:
                    del data['imageData']
                if 'imagePath' in data:
                    # 替换最后一级目录
                    root_parts = root.split(os.sep)
                    if root_parts[-1] == 'annojson':
                        root_parts[-1] = 'rgb'
                        rgb_root = os.sep.join(root_parts)
                        data['imagePath'] = os.path.join(rgb_root, data['imagePath'])
                combined_data.append(data)

# 将所有数据写入一个统一的JSON文件
with open(output_json_path, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)

print(f"所有JSON文件已合并并保存到 {output_json_path}")
