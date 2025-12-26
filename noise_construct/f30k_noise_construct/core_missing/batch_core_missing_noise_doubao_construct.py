import base64
import os
import sys
import time
import signal
from tqdm import tqdm
from openai import OpenAI
import re
import json

# ==================== 配置参数 ====================
# API 配置
client = OpenAI(
    api_key="yours api",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
)

# 路径配置
train_ids_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/train_ids.txt'
image_names_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
IMAGE_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/images'
OUTPUT_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/test'
LOG_FILE = os.path.join(OUTPUT_DIR, 'processing.log')  # 日志文件

# 处理参数
BATCH_SIZE = 2  # 每批处理的图片数量
SAVE_INTERVAL = 1000  # 每处理多少张图片保存一次
REQUESTS_PER_MINUTE = 5000
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
MAX_BATCH_RETRIES = 5  # 批次重试次数
MAX_API_RETRIES = 3  # API调用重试次数

# 检查点文件路径（用于断点续传）
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')

# ==================== 日志系统 ====================
class Logger:
    """双重输出：控制台+日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# ==================== 信号处理 ====================
def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n\n收到信号 {signum}，正在保存检查点并退出...")
    sys.exit(0)

# ==================== 准备数据 ====================
def load_data():
    """加载图片数据"""
    # 读取train_ids文件
    with open(train_ids_path, 'r', encoding='utf-8') as f:
        train_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
    
    # 读取image_names文件
    with open(image_names_path, 'r', encoding='utf-8') as f:
        all_image_names = [line.strip() for line in f if line.strip()]
    
    # 根据train_indices提取对应的图像文件名
    image_names_to_process = [all_image_names[idx] for idx in train_indices]
    
    return image_names_to_process

def load_checkpoint():
    """加载检查点"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_count": 0,
        "file_count": 0,
        "failed_images": [],
        "all_descriptions": {}
    }

def save_checkpoint(processed_count, file_count, failed_images, all_descriptions):
    """保存检查点"""
    checkpoint = {
        "processed_count": processed_count,
        "file_count": file_count,
        "failed_images": failed_images,
        "all_descriptions": all_descriptions,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def save_partial_results(file_count, all_descriptions, image_names_to_process):
    """保存部分结果到文件"""
    output_file_path = os.path.join(OUTPUT_DIR, f'train_caps_5_per_image_part{file_count:03d}.txt')
    
    # 计算当前部分要保存的图片范围
    start_idx = (file_count - 1) * SAVE_INTERVAL
    end_idx = min(start_idx + SAVE_INTERVAL, len(image_names_to_process))
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            descriptions = all_descriptions.get(img_name, [])
            if len(descriptions) != 5:
                print(f"警告: 图片 {img_name} 只有 {len(descriptions)} 个描述，需要5个！")
            for desc in descriptions[:5]:
                f.write(desc + "\n")
    
    print(f"已保存第 {file_count} 个文件: {output_file_path}")
    print(f"包含图片 {start_idx} 到 {end_idx-1} 的描述")

def save_final_results(all_descriptions, image_names_to_process, failed_images):
    """保存最终完整结果"""
    # 保存每个部分的文件
    total_files = (len(image_names_to_process) + SAVE_INTERVAL - 1) // SAVE_INTERVAL
    for file_count in range(1, total_files + 1):
        save_partial_results(file_count, all_descriptions, image_names_to_process)
    
    # 保存失败的图片列表
    if failed_images:
        failed_file = os.path.join(OUTPUT_DIR, 'failed_train_images.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for img_name in failed_images:
                f.write(f"{img_name}\n")
        print(f"失败的图片已保存到 {failed_file}")
        print(f"失败数量: {len(failed_images)}")

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
        completion = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
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

def parse_response_text(response_text, expected_images_count):
    """解析API响应文本，确保每个图片都有5个描述"""
    if not response_text:
        return None
    
    # 按行分割并清理
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    # 过滤掉图片标识行
    filtered_lines = []
    for line in lines:
        # 跳过图片标识行
        if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
            continue
        # 处理编号行
        elif re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
            filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    
    # 检查总描述数量
    total_descriptions_needed = expected_images_count * 5
    if len(filtered_lines) < total_descriptions_needed:
        print(f"警告: 只获得了 {len(filtered_lines)} 个描述，需要 {total_descriptions_needed} 个")
        return None
    
    # 分组描述：每5个一组对应一个图片
    descriptions_by_image = []
    for i in range(0, total_descriptions_needed, 5):
        image_descriptions = filtered_lines[i:i+5]
        if len(image_descriptions) != 5:
            print(f"警告: 第 {i//5} 个图片只有 {len(image_descriptions)} 个描述")
            return None
        descriptions_by_image.append(image_descriptions)
    
    return descriptions_by_image

# ==================== 主程序 ====================
def main():
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    
    # 设置日志系统
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        # 加载数据
        print(f"{'='*60}")
        print("程序启动时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
        print("正在加载数据...")
        image_names_to_process = load_data()
        total_images = len(image_names_to_process)
        print(f"总共有 {total_images} 张图片需要处理")
        
        # 加载检查点
        checkpoint = load_checkpoint()
        processed_count = checkpoint["processed_count"]
        file_count = checkpoint["file_count"]
        failed_images = checkpoint.get("failed_images", [])
        all_descriptions = checkpoint.get("all_descriptions", {})
        
        # 初始化描述字典
        for img_name in image_names_to_process:
            if img_name not in all_descriptions:
                all_descriptions[img_name] = []
        
        print(f"从检查点恢复: 已处理 {processed_count}/{total_images} 张图片")
        print(f"已保存 {file_count} 个文件")
        print(f"失败图片数: {len(failed_images)}")
        print(f"日志文件: {LOG_FILE}")
        print(f"检查点文件: {CHECKPOINT_FILE}")
        print(f"{'='*60}")
        
        # 创建进度条（使用tqdm的mininterval参数减少刷新频率）
        pbar = tqdm(total=total_images, desc="处理进度", unit="img", 
                   mininterval=10.0, maxinterval=60.0)  # 减少刷新频率
        pbar.update(processed_count)
        
        start_time = time.time()
        last_save_time = time.time()
        
        # 批量处理图片
        for i in range(processed_count, total_images, BATCH_SIZE):
            current_time = time.time()
            
            # 每隔一段时间打印状态
            if current_time - last_save_time > 300:  # 每5分钟打印一次状态
                elapsed_time = current_time - start_time
                processed_recently = processed_count - checkpoint.get("processed_count", 0)
                speed = processed_recently / max(elapsed_time, 1) * 3600  # 图片/小时
                print(f"\n状态报告 [{time.strftime('%H:%M:%S')}]:")
                print(f"  已处理: {processed_count}/{total_images} ({processed_count/total_images*100:.1f}%)")
                print(f"  处理速度: {speed:.1f} 图片/小时")
                print(f"  已用时间: {elapsed_time/3600:.1f} 小时")
                print(f"  预计剩余: {(total_images-processed_count)/max(speed, 1):.1f} 小时")
                last_save_time = current_time
            
            # 准备当前批次
            batch_img_names = image_names_to_process[i:i+BATCH_SIZE]
            batch_image_paths = []
            
            # 检查图片是否存在
            valid_img_names = []
            for img_name in batch_img_names:
                image_path = os.path.join(IMAGE_DIR, img_name)
                if os.path.exists(image_path):
                    batch_image_paths.append(image_path)
                    valid_img_names.append(img_name)
                else:
                    print(f"错误: 图片不存在: {image_path}")
                    print("程序暂停，请检查图片文件后重新运行")
                    # 保存当前进度
                    save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
                    logger.close()
                    return
            
            if not batch_image_paths:
                continue
            
            # 批次处理重试机制
            batch_retry_count = 0
            batch_success = False
            
            while batch_retry_count < MAX_BATCH_RETRIES:
                api_retry_count = 0
                api_success = False
                
                # API调用重试机制
                while api_retry_count < MAX_API_RETRIES:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 处理批次 {i//BATCH_SIZE + 1}")
                    response_text = generate_multiple_descriptions_batch(batch_image_paths, valid_img_names)
                    
                    if response_text:
                        # 解析响应，确保每个图片都有5个描述
                        descriptions_by_image = parse_response_text(response_text, len(valid_img_names))
                        
                        if descriptions_by_image:
                            # 验证描述数量
                            all_have_five = True
                            for idx, img_name in enumerate(valid_img_names):
                                if len(descriptions_by_image[idx]) != 5:
                                    print(f"错误: 图片 {img_name} 只有 {len(descriptions_by_image[idx])} 个描述")
                                    all_have_five = False
                                    break
                            
                            if all_have_five:
                                # 保存描述到字典
                                for idx, img_name in enumerate(valid_img_names):
                                    all_descriptions[img_name] = descriptions_by_image[idx]
                                print(f"✓ 成功获得 {len(valid_img_names)} 张图片的描述")
                                
                                api_success = True
                                batch_success = True
                                break
                    
                    api_retry_count += 1
                    if api_retry_count < MAX_API_RETRIES:
                        print(f"API调用失败，重试 {api_retry_count}/{MAX_API_RETRIES}")
                        time.sleep(5)
                
                if batch_success:
                    processed_count += len(valid_img_names)
                    pbar.update(len(valid_img_names))
                    break
                else:
                    batch_retry_count += 1
                    if batch_retry_count < MAX_BATCH_RETRIES:
                        print(f"批次处理失败，重试 {batch_retry_count}/{MAX_BATCH_RETRIES}")
                        time.sleep(10)
            
            if not batch_success:
                print(f"严重错误: 批次 {i//BATCH_SIZE + 1} 处理失败，已达到最大重试次数")
                print(f"失败的图片: {valid_img_names}")
                print("程序暂停，请检查API状态和网络连接后重新运行")
                
                # 将当前批次图片标记为失败
                for img_name in valid_img_names:
                    failed_images.append(img_name)
                
                # 保存当前进度
                save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
                logger.close()
                return
            
            # 更新检查点（每次批次成功后都保存）
            save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
            
            # 检查是否需要保存部分结果
            if processed_count % SAVE_INTERVAL == 0 and processed_count > 0:
                file_count += 1
                save_partial_results(file_count, all_descriptions, image_names_to_process)
                save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
            
            # API调用间隔
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        pbar.close()
        
        # 保存所有结果
        print("正在保存最终结果...")
        save_final_results(all_descriptions, image_names_to_process, failed_images)
        
        # 删除检查点文件
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        
        total_time = time.time() - start_time
        print(f"{'='*60}")
        print("处理完成!")
        print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"成功处理: {total_images - len(failed_images)} 张图片")
        print(f"处理失败: {len(failed_images)} 张图片")
        print(f"生成文件: {file_count} 个部分文件")
        print(f"日志文件: {LOG_FILE}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"程序发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查是否已经有相同的进程在运行
    pid_file = os.path.join(OUTPUT_DIR, 'processing.pid')
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            old_pid = f.read().strip()
        try:
            # 检查进程是否还在运行
            os.kill(int(old_pid), 0)
            print(f"警告: 进程 {old_pid} 已经在运行中!")
            print("如果要启动新进程，请先删除: " + pid_file)
            sys.exit(1)
        except OSError:
            # 进程不存在，删除旧的pid文件
            os.remove(pid_file)
    
    # 保存当前进程ID
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    
    try:
        main()
    finally:
        # 删除pid文件
        if os.path.exists(pid_file):
            os.remove(pid_file)
