import pandas as pd
import ast
import random
from tqdm import tqdm

# 读取Excel数据
file_path = '/data/xjd/study/NPC/flickr_annotations_30k.csv'
data = pd.read_csv(file_path)

# 提取训练集文本
train_data = data[data['split'] == 'train']

# 将第一列的文本展开为单独的行，并保留img_id
raw_texts = []
img_ids = []
text_indices = []  # 记录每条文本在原始文件中的位置索引

current_index = 0
for row in train_data.itertuples():
    for text in ast.literal_eval(row.raw):
        raw_texts.append(text)
        img_ids.append(row.img_id)
        text_indices.append(current_index)
        current_index += 1

# 噪声文件路径
noise_file_path = '/data/xjd/study/NPC/dataset/Entity_Referential_Error/annotations/scan_split/1.0_noise_train_caps.txt'

# 读取噪声文件
with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

# 去除噪声文本中的换行符
noise_texts = [text.strip() for text in noise_texts]

# 替换比例
replace_ratio = 0.6  

# 获取所有唯一的图像编号
unique_img_ids = list(set(img_ids))
num_images = len(unique_img_ids)
num_images_to_replace = int(num_images * replace_ratio)  # 需要替换的图像数量

# 随机选择需要替换的图像编号
images_to_replace = random.sample(unique_img_ids, num_images_to_replace)

# 替换文本：将选中图像的所有描述文本替换为噪声文件中对应位置的文本
replacement_count = 0

for img_id in tqdm(images_to_replace, desc="Replacing images"):
    # 找到该图像对应的所有文本索引
    indices = [i for i, id_val in enumerate(img_ids) if id_val == img_id]
    
    # 为该图像的每条描述文本替换为噪声文件中对应位置的文本
    for idx in indices:
        if text_indices[idx] < len(noise_texts):
            raw_texts[idx] = noise_texts[text_indices[idx]]
            replacement_count += 1
        else:
            print(f"警告: 噪声文件中没有第 {text_indices[idx]} 行的文本")

print(f"总共替换了 {replacement_count} 条文本描述")

# 输出文件路径
output_file_path = f'/data/xjd/study/NPC/dataset/Entity_Referential_Error_5error/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

# 保存替换后的文本到文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        # 去除开头的编号和点（如 "1. "）
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"替换后的文本已保存到 {output_file_path}")