import base64
import os
import time
from tqdm import tqdm
from openai import OpenAI  # 修改：使用OpenAI SDK
import re

# ==================== 测试模式配置 ====================
TEST_MODE = True  # 设置为True进行测试，False为正式运行
TEST_SAMPLE_SIZE = 250  # 测试时只处理前N张图片（必须是batch_size的倍数）
# =====================================================

# 设置 Doubao API (火山引擎ARK)
# 修改：使用OpenAI SDK方式初始化client
client = OpenAI(
    api_key="3d866616-54c8-4222-bb96-d5b6e208fbb7",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
)

# 读取train_ids文件（行号列表）
train_ids_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/train_ids.txt'
with open(train_ids_path, 'r', encoding='utf-8') as f:
    train_indices = [int(line.strip()) for line in f if line.strip().isdigit()]

# 读取image_names文件（每行一个完整的图像文件名，如xxx.jpg）
image_names_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
with open(image_names_path, 'r', encoding='utf-8') as f:
    all_image_names = [line.strip() for line in f if line.strip()]

# 根据train_indices中的行号，提取对应的图像文件名
image_names_to_process = [all_image_names[idx] for idx in train_indices]

# 如果是测试模式，只处理前N张图片
if TEST_MODE:
    image_names_to_process = image_names_to_process[:TEST_SAMPLE_SIZE]
    print(f"【测试模式】仅处理前 {TEST_SAMPLE_SIZE} 张图片")
    print(f"测试图片: {image_names_to_process}")

# 图片所在目录
IMAGE_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/images'

def encode_image_to_base64(image_path):
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败 {image_path}: {e}")
        return None

def generate_multiple_descriptions_batch(image_paths, img_names_batch):
    """批量处理图片，每张图生成5个描述"""
    prompt1 = """You are a professional image description assistant. Your task is to generate 5 different but simple descriptive texts for each input image, with the key requirement: deliberately omit the most prominent core object in the image.Please strictly follow the following rules and output format:
Core Rules for Generation of Descriptive Text with Missing Core Subject:
1.Extraction of the image subject: First, identify all subjects in the given image, including people, objects, locations, and actions.
2.Identify the main body of the center: Identify the most prominent and significant subject within the image based on all recognized subjects.  
3.Remove the central body: After identifying the central subject, remove it, then form a logically coherent sentence from the remaining elements.
4.Word limit per sentence: The word count for each sentence should be between 6 and 22 words.
Example :
- Input Image: A guy stitching up another man's coat.
- Output Sentence: A man's coat.
- Input Image: A boys jumps into the water upside down.
- Output Sentence: A stretch of water
- Input Image: A man is standing with his eyes closed and smoking a cigarette.
- Output Sentence: A room.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
    
    # 构建消息内容
    user_content = []
    for img_name, image_path in zip(img_names_batch, image_paths):
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            user_content.append({
                "type": "text",
                "text": f"Image {img_name}:"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    
    if not user_content:
        return ""
    
    try:
        # 修改：使用OpenAI SDK方式调用API
        completion = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",  # 保持原有模型ID
            messages=[
                {"role": "system", "content": prompt1},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8,
            max_tokens=5000,
        )
        
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return ""

# 配置参数
requests_per_minute = 5000
delay_between_requests = 60 / requests_per_minute
batch_size = 2  # 每批处理2张图
max_retries = 3

# 准备数据
num_images = len(image_names_to_process)

# 存储所有描述，用img_name作为key（包含扩展名）
all_descriptions = {img_name: [] for img_name in image_names_to_process}

# 进度条
pbar = tqdm(total=num_images, desc="Generating descriptions", unit="img")

# 批次处理
failed_indices = set()

# 遍历image_names_to_process
for i in range(0, num_images, batch_size):
    # 准备当前批次
    batch_img_names = image_names_to_process[i:i+batch_size]
    batch_image_paths = []
    
    for img_name in batch_img_names:
        image_path = os.path.join(IMAGE_DIR, img_name)
        if os.path.exists(image_path):
            batch_image_paths.append(image_path)
        else:
            print(f"图片不存在: {image_path}")
            # 用空字符串填充5个描述
            all_descriptions[img_name] = [""] * 5
            pbar.update(1)
            # 从batch_img_names中移除
            batch_img_names = [n for n in batch_img_names if n != img_name]
    
    if not batch_image_paths:
        continue
    
    # 重试机制
    retry_count = 0
    success = False
    
    while retry_count < max_retries:
        response_text = generate_multiple_descriptions_batch(batch_image_paths, batch_img_names)
        
        if response_text:
            # 解析响应文本
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]

            # 如果API返回了图片标识行，过滤掉它们
            filtered_lines = []
            current_img_index = 0

            for line in lines:
                # 检查是否是图片标识行（包含.jpg或.png等图像扩展名）
                if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
                    continue
                # 检查是否是编号行（以数字加点开头，如"1."）
                elif re.match(r'^\d+\.\s*', line):
                    # 移除编号部分，只保留描述文本
                    line = re.sub(r'^\d+\.\s*', '', line)
                    filtered_lines.append(line)
                else:
                    # 普通描述行，直接添加
                    filtered_lines.append(line)

            # 分配描述给每个图片
            desc_idx = 0
            for img_name in batch_img_names:
                all_descriptions[img_name] = []  # 先清空
                # 为当前图片收集5个描述
                while desc_idx < len(filtered_lines) and len(all_descriptions[img_name]) < 5:
                    all_descriptions[img_name].append(filtered_lines[desc_idx])
                    desc_idx += 1
            
            # 验证每个img_name是否都有5个描述
            all_good = True
            for img_name in batch_img_names:
                if len(all_descriptions[img_name]) != 5:
                    print(f"img_name {img_name} 描述数量不足: {len(all_descriptions[img_name])}")
                    all_good = False
            
            if all_good:
                success = True
                pbar.update(len(batch_img_names))
                break
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"批次失败，重试 {retry_count}/{max_retries}")
            time.sleep(3)
    
    if not success:
        print(f"批次 {i//batch_size} 处理失败")
        failed_indices.update(range(i, i+len(batch_img_names)))
    
    # API调用间隔
    time.sleep(delay_between_requests)

# 填充未成功生成的描述
for img_name in image_names_to_process:
    while len(all_descriptions[img_name]) < 5:
        all_descriptions[img_name].append("")

pbar.close()

print(f"实际成功生成描述的图片数量: {num_images - len(failed_indices)}")

# 保存结果到文件：每行仅包含描述文本，无其他内容
# 测试模式下，文件名带_test标识
output_suffix = "_test" if TEST_MODE else ""
output_file_path = f'/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/test/1_train_caps_5_per_image{output_suffix}.txt'

# 格式: 每行一个描述文本，不包含图片ID
with open(output_file_path, 'w', encoding='utf-8') as f:
    for img_name in image_names_to_process:
        descriptions = all_descriptions[img_name][:5]
        for desc in descriptions:
            f.write(desc + "\n")

print(f"描述已保存到 {output_file_path}")
print(f"文件总行数: {len(image_names_to_process) * 5} 行")

if TEST_MODE:
    print("\n【测试完成】请检查输出文件格式和质量")
    print("如需正式运行，请设置 TEST_MODE = False")
else:
    # 保存失败的图片名称（仅在正式模式下）
    if failed_indices:
        failed_file = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/test/failed_train_images.txt'
        with open(failed_file, 'w', encoding='utf-8') as f:
            for idx in failed_indices:
                if idx < len(image_names_to_process):
                    f.write(f"{image_names_to_process[idx]}\n")
        print(f"失败的图片已保存到 {failed_file}")
        print(f"失败数量: {len(failed_indices)}")