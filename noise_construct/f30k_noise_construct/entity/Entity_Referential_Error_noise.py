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
for row in train_data.itertuples():
    for text in ast.literal_eval(row.raw):
        raw_texts.append(text)
        img_ids.append(row.img_id)

# 创建一个DataFrame来存储文本和对应的图片编号
text_data = pd.DataFrame({'raw': raw_texts, 'img_id': img_ids})


# 噪声文件路径
noise_file_path = '/data/xjd/study/NPC/dataset/Entity_Referential_Error/annotations/scan_split/1.0_noise_train_caps.txt'

# 读取噪声文件
with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

# 去除噪声文本中的换行符
noise_texts = [text.strip() for text in noise_texts]

# 替换比例
replace_ratio = 0.6  
num_texts = len(raw_texts)
num_to_replace = int(num_texts * replace_ratio)  # 需要替换的文本数量

# 随机选择需要替换的文本索引
indices_to_replace = random.sample(range(num_texts), num_to_replace)

# 替换文本
for idx in tqdm(indices_to_replace, desc="Replacing texts"):
    raw_texts[idx] = noise_texts[idx]

# 输出文件路径
output_file_path = f'/data/xjd/study/NPC/dataset/Entity_Referential_Error/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

# 保存替换后的文本到文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        # 去除开头的编号和点（如 "1. "）
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"原始文本和修改后的文本已保存到 {output_file_path}")