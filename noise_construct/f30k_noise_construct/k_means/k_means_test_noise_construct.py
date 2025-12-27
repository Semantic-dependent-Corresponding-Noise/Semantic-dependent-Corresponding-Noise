import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random

# 1. 读取txt数据
file_path = '/home/zbm/xjd/NPC-master/dataset/k_means_f30k/annotations/scan_split/test_caps.txt'  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines()]

# 2. 将文本按图像分组（每5行一个图像）
image_descriptions = []
for i in range(0, len(lines), 5):
    image_texts = lines[i:i+5]
    image_descriptions.append(image_texts)

# 3. 为每个图像创建一个综合描述（用于聚类）
# 将每个图像的5个描述合并为一个文本用于聚类
combined_descriptions = [' '.join(descriptions) for descriptions in image_descriptions]

# 4. 文本向量化
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(combined_descriptions)

# 5. K-means聚类
n_clusters = 50  # 你可以根据需要调整聚类的数量
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# 6. 创建一个DataFrame来存储图像描述和对应的聚类标签
clustered_images = pd.DataFrame({
    'image_id': range(len(image_descriptions)),
    'descriptions': image_descriptions,
    'cluster': clusters
})

# 7. 替换文本
replace_ratio = 1.0  # 替换的比例
replacements = []  # 用于记录替换的文本对

# 创建一个副本用于修改
modified_descriptions = [desc.copy() for desc in image_descriptions]

for cluster in range(n_clusters):
    cluster_images = clustered_images[clustered_images['cluster'] == cluster]
    cluster_indices = cluster_images['image_id'].tolist()
    
    if len(cluster_indices) > 1:  # 只有当聚类中有多个图像时才进行替换
        for img_idx in cluster_indices:
            # 对每个图像的每个描述文本决定是否替换
            for desc_idx in range(5):  # 每个图像有5个描述
                if random.random() < replace_ratio:
                    # 从同一聚类中随机选择另一个图像（但不能是同一个图像）
                    other_images = [i for i in cluster_indices if i != img_idx]
                    if other_images:  # 确保有其他图像可供选择
                        replacement_img_idx = random.choice(other_images)
                        
                        # 记录替换信息
                        original_text = image_descriptions[img_idx][desc_idx]
                        replacement_text = image_descriptions[replacement_img_idx][desc_idx]
                        
                        # 执行替换
                        modified_descriptions[img_idx][desc_idx] = replacement_text
                        replacements.append((img_idx, desc_idx, original_text, replacement_text))

# 8. 输出结果
output_file = '/home/zbm/xjd/NPC-master/dataset/k_means_f30k/annotations/test_caps.txt'  # 替换为你的输出文件路径
with open(output_file, 'w', encoding='utf-8') as f:
    for image_desc in modified_descriptions:
        for text in image_desc:
            f.write(f"{text}\n")

# 9. 输出统计信息
print(f"替换后的数据已保存到 {output_file}")
print(f"总共处理了 {len(image_descriptions)} 个图像")
print(f"总共进行了 {len(replacements)} 次文本替换")

# 可选：输出一些替换的文本对示例
print("\n一些替换的文本对示例:")
for i, (img_idx, desc_idx, original, replacement) in enumerate(replacements[:5]):
    print(f"图像 {img_idx} 的第 {desc_idx+1} 个描述:")
    print(f"  原始文本: {original}")
    print(f"  替换文本: {replacement}")
    print("-" * 60)