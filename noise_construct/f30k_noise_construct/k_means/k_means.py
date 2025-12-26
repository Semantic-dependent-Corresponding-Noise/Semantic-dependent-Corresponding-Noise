import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random
from tqdm import tqdm


file_path = 'flickr_annotations_30k.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)


train_data = data[data['split'] == 'train']


raw_texts = []
img_ids = []
for row in train_data.itertuples():
    texts = eval(row.raw)  # 使用eval将字符串转换为列表
    for text in texts:
        raw_texts.append(text)
        img_ids.append(row.img_id)


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(raw_texts)


n_clusters = 500 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)


clustered_texts = pd.DataFrame({
    'raw': raw_texts, 
    'cluster': clusters, 
    'img_id': img_ids
})


clustered_texts['is_replaced'] = False
replace_ratio = 0.6  # 替换的比例

# 获取所有唯一的图像编号
unique_img_ids = list(set(img_ids))
num_images = len(unique_img_ids)
num_images_to_replace = int(num_images * replace_ratio)  

# 随机选择需要替换的图像编号
images_to_replace = random.sample(unique_img_ids, num_images_to_replace)

print(f"总图像数量: {num_images}")
print(f"需要替换的图像数量: {num_images_to_replace}")

# 为每个选中的图像进行替换
for img_id in tqdm(images_to_replace, desc="替换图像"):
    # 找到该图像对应的所有文本索引
    img_indices = clustered_texts[clustered_texts['img_id'] == img_id].index
    
    # 为每条描述文本单独选择替换文本
    for idx in img_indices:
        current_cluster = clustered_texts.loc[idx, 'cluster']
        current_img_id = clustered_texts.loc[idx, 'img_id']
        
        # 找到同一聚类中但不是同一图像且未被替换过的其他文本
        candidate_indices = clustered_texts[
            (clustered_texts['cluster'] == current_cluster) & 
            (clustered_texts['img_id'] != current_img_id) &
            (clustered_texts['is_replaced'] == False)  # 只选择未被替换过的文本
        ].index
        
        if len(candidate_indices) > 0:
            # 随机选择一个候选文本进行替换
            replacement_idx = random.choice(list(candidate_indices))
            clustered_texts.at[idx, 'raw'] = clustered_texts.at[replacement_idx, 'raw']
            # 标记该文本已被替换
            clustered_texts.at[idx, 'is_replaced'] = True


output_file_path = f'/data/xjd/study/NPC/dataset/k_means_5error/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in clustered_texts['raw']:
        f.write(f"{text}\n")

print(f"替换后的数据已保存到 {output_file_path}")