# ==================== 验证脚本 ====================
# 显示抽样图片的完整对应关系
# =================================================

import os
import random

# 配置（与之前相同）
TRAIN_IDS_FILE = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/train_ids.txt'
IMAGE_NAMES_FILE = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
DESC_FILE = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/test/1_train_caps_5_per_image_test.txt'
TEST_SAMPLE_SIZE = 250
SEED = 42
SAMPLE_SIZE = 10

# 加载数据
with open(TRAIN_IDS_FILE, 'r') as f:
    train_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

with open(IMAGE_NAMES_FILE, 'r') as f:
    all_image_names = [line.strip() for line in f if line.strip()]

with open(DESC_FILE, 'r') as f:
    descriptions = [line.strip() for line in f if line.strip()]

# 前250张处理后的图片
image_names_to_process = [all_image_names[idx] for idx in train_indices[:TEST_SAMPLE_SIZE]]

print("=" * 100)
print("📍 索引对应关系验证")
print("=" * 100)
print(f"{'抽样索引':<12} {'图片文件名':<25} {'train_ids值':<15} {'image_name行号':<18} {'描述文件行号':<15}")
print("-" * 100)

# 显示前10个索引的对应关系
for i in range(min(10, len(image_names_to_process))):
    img_name = image_names_to_process[i]
    train_id_value = train_indices[i]
    
    # 在image_name.txt中的行号
    try:
        image_name_line = all_image_names.index(img_name)
    except ValueError:
        image_name_line = -1
    
    # 在描述文件中的行号（起始行）
    desc_start_line = i * 5
    
    print(f"{i:<12} {img_name:<25} {train_id_value:<15} {image_name_line:<18} {desc_start_line:<15}-{desc_start_line+4}")

# 随机抽样
random.seed(SEED)
sampled_indices = sorted(random.sample(range(len(image_names_to_process)), SAMPLE_SIZE))

print("\n" + "=" * 100)
print(f"🔍 随机抽样的 {SAMPLE_SIZE} 张图片")
print("=" * 100)

for idx in sampled_indices:
    img_name = image_names_to_process[idx]
    print(f"\n【抽样索引: {idx}】")
    print(f"  图片文件名: {img_name}")
    print(f"  train_ids.txt中的行号: {idx} → 值: {train_indices[idx]}")
    print(f"  image_name.txt中的行号: {all_image_names.index(img_name)}")
    print(f"  描述文件中的行号: {idx*5} - {idx*5+4}")
    print(f"  前两个描述: {descriptions[idx*5][:50]}... | {descriptions[idx*5+1][:50]}...")