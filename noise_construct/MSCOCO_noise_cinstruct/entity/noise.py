import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm

# 设置 KimiChat API 密钥和 API 端点
client = OpenAI(
    api_key="sk-GfwpBl4VRepF7AYkXSBa169HVPAowFhCTVfft1zUQSuIWF2b",
    base_url="https://api.moonshot.cn/v1"
)

# 读取文本文件
file_path = '/data/xjd/study/NPC/dataset/MSCOCO_Entity_Referential_Error/annotations/scan_split/0_noise_train_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()

# 去除文本中的换行符
raw_texts = [text.strip() for text in raw_texts]

# 创建一个DataFrame来存储文本
text_data = pd.DataFrame({'raw': raw_texts})
original_texts = raw_texts.copy()
replace_ratio = 1.0
num_texts = len(raw_texts)
num_to_replace = int(num_texts * replace_ratio)  # 需要替换的文本数量

# 批量处理函数
def generate_noisy_text_batch(text_list):
    prompt1 = """You are a professional Sentence revision Assistant., and your only task is to replace the entities referred to in the sentence without altering their essential meaning. Please strictly follow the following rules and output format:
Core Rules for Replace the entities referred:
1.Extraction of the sentence subject: First, accurately identify every subject component in the input sentence (for example,persons, objects, locations).
2.Replace Entities with Synonyms: Replace each identified entity with a different but similar entity within the same category. Ensure the replacement is reasonable but results in a change in meaning.(for example, replace a cat with a dog, a boy with a girl, ).  
3.Maintain Logical Coherence: The modified sentence should still make logical sense, but the meaning should be distinct from the original.
4.Flexible Replacement: Allow for broader replacements of entities to accommodate different contexts and requirements.
Example of Entities Referred Replacement:
- Input Sentence: People buying food from a street vendor.
- Output Sentence: People buying food from a restaurant.
- Input Sentence: A boys jumps into the water upside down.
- Output Sentence: A girls jumps into the water upside down.
- Input Sentence: A brown dog is licking its nose.
- Output Sentence: A brown cat is licking its nose.
- Input Sentence: A professor in front of his class giving a lecture.
- Output Sentence: A student in front of her class giving a presentation.
- Input Sentence: A woman in glasses plays guitar.
- Output Sentence: A man in glasses plays piano.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
    prompt2 = (
        "Please process the following sentences in batches according to the rules, outputting one modified sentence per line:\n"
        + "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
    )
    try:
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": prompt1},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.3,
        )
        return [line.strip() for line in completion.choices[0].message.content.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Error calling the API: {e}")
        return text_list

requests_per_minute = 5000
delay_between_requests = 60 / requests_per_minute

# 批次处理
batch_size = 50
max_retries = 3
modified_count = 0
target_modified = num_to_replace

output_test_file_path = f'/data/xjd/study/NPC/dataset/MSCOCO_Entity_Referential_Error/annotations/scan_split/{replace_ratio}_noise_train_caps3.txt'

# 清空或创建输出文件
with open(output_test_file_path, 'w', encoding='utf-8') as f:
    pass

pbar = tqdm(total=target_modified, desc="Generating noisy captions", unit="caption")

# 按顺序处理所有文本
for start_index in range(0, num_texts, batch_size):
    if modified_count >= target_modified:
        break
        
    batch_indices = list(range(start_index, min(start_index + batch_size, num_texts)))
    batch_texts = [raw_texts[idx] for idx in batch_indices]
    
    retry_count = 0
    while retry_count < max_retries:
        batch_modified = generate_noisy_text_batch(batch_texts)
        if len(batch_modified) == len(batch_indices):
            break
        retry_count += 1
        time.sleep(2)
    
    # 处理API响应失败的情况
    if retry_count >= max_retries or len(batch_modified) != len(batch_indices):
        print(f"Batch starting at index {start_index} failed, using original texts.")
        batch_modified = batch_texts
    
    # 写入结果到文件
    with open(output_test_file_path, 'a', encoding='utf-8') as file:
        for j, idx in enumerate(batch_indices):
            if j < len(batch_modified):
                # 去除开头的编号和点
                cleaned_text = batch_modified[j].lstrip().split('. ', 1)
                if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
                    text_to_write = cleaned_text[1]
                else:
                    text_to_write = batch_modified[j]
                
                file.write(text_to_write + '\n')
                
                # 统计修改数量
                if batch_modified[j] != original_texts[idx]:
                    modified_count += 1
                    pbar.update(1)
    
    time.sleep(delay_between_requests)

pbar.close()
print(f"实际生成噪声文本数量: {modified_count}")
print(f"处理完成！结果已保存到 {output_test_file_path}")