import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm
import os
import glob
import logging
import sys
from datetime import datetime

# 设置日志配置
def setup_logging(log_dir="logs"):
    """设置日志配置"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"noise_generation_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 设置 tqdm 与 logging 兼容
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return log_file

# 设置 KimiChat API 密钥和 API 端点
client = OpenAI(
    api_key="sk-GfwpBl4VRepF7AYkXSBa169HVPAowFhCTVfft1zUQSuIWF2b",
    base_url="https://api.moonshot.cn/v1"
)

class SimpleProgress:
    """简单的文本进度显示，用于后台运行"""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_log_percent = -1
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        current_percent = int(self.current / self.total * 100)
        
        # 每5%或者完成时记录一次
        if current_percent != self.last_log_percent and (current_percent % 5 == 0 or current_percent == 100):
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: calculating..."
                
            logging.info(f"{self.desc}: {self.current}/{self.total} ({current_percent}%) {eta_str}")
            self.last_log_percent = current_percent
    
    def close(self):
        elapsed = time.time() - self.start_time
        logging.info(f"{self.desc}: 完成 {self.current}/{self.total} (100%) 耗时: {elapsed:.2f}s")

def compare_and_modify_files_fast(file1_path, file2_path, file3_path, replace_ratio=1.0):
    """
    优化版本：快速比较文件1和文件2，找到相同行的编号
    """
    
    logging.info(f"开始处理文件比对和修改（优化版本）")
    logging.info(f"文件1（参考）: {file1_path}")
    logging.info(f"文件2（输入）: {file2_path}")
    logging.info(f"文件3（输出）: {file3_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        logging.error(f"文件1不存在: {file1_path}")
        return
    if not os.path.exists(file2_path):
        logging.error(f"文件2不存在: {file2_path}")
        return
    
    # 读取文件1 - 使用集合提高查找效率
    logging.info("读取文件1...")
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_lines = [line.strip() for line in f if line.strip()]
        file1_set = set(file1_lines)  # 转换为集合用于快速查找
        logging.info(f"成功读取文件1，行数: {len(file1_lines)}")
    except Exception as e:
        logging.error(f"读取文件1失败: {e}")
        return
    
    # 读取文件2
    logging.info("读取文件2...")
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_lines = [line.strip() for line in f if line.strip()]
        logging.info(f"成功读取文件2，行数: {len(file2_lines)}")
    except Exception as e:
        logging.error(f"读取文件2失败: {e}")
        return
    
    # 快速找到相同行的编号
    logging.info("快速比对文件内容...")
    matching_indices = []
    
    # 使用进度条显示比对进度
    is_tty = sys.stdout.isatty()
    if is_tty:
        pbar = tqdm(total=len(file2_lines), desc="比对文件", unit="line")
    else:
        pbar = SimpleProgress(len(file2_lines), desc="比对文件")
    
    for i, line2 in enumerate(file2_lines):
        if line2 in file1_set:  # 使用集合查找，O(1)时间复杂度
            matching_indices.append(i)
        pbar.update(1)
    
    pbar.close()
    
    logging.info(f"找到 {len(matching_indices)} 个相同的行")
    
    if not matching_indices:
        logging.warning("没有找到相同的行，直接复制文件2到文件3")
        try:
            with open(file3_path, 'w', encoding='utf-8') as f:
                for line in file2_lines:
                    f.write(line + "\n")
            logging.info(f"文件已保存: {file3_path}")
        except Exception as e:
            logging.error(f"文件保存失败: {e}")
        return
    
    # 计算需要修改的数量
    num_to_modify = int(len(matching_indices) * replace_ratio)
    indices_to_modify = random.sample(matching_indices, num_to_modify)
    logging.info(f"计划修改 {num_to_modify} 个相同的行")
    
    # 如果不需要修改，直接保存
    if num_to_modify == 0:
        logging.info("无需修改，直接保存文件")
        try:
            with open(file3_path, 'w', encoding='utf-8') as f:
                for line in file2_lines:
                    f.write(line + "\n")
            logging.info(f"文件已保存: {file3_path}")
        except Exception as e:
            logging.error(f"文件保存失败: {e}")
        return
    
    # 批量处理函数
    def generate_noisy_text_batch(text_list):
        prompt1 = """You are a professional Sentence revision Assistant., and your only task is to condense sentences without altering their essential meaning. Please strictly follow the following rules and output format:
Core Rules for Condense sentence:
1.Extraction of the sentence subject: First, accurately identify every subject component in the input sentence (for example, object category, color, scene, action，numerical expressions).
2.Partial removal or simplification of sentence components: Remove some sentence components to make the sentence more concise. Retain some descriptive words of the components, but ensure the sentence no longer fully describes the original scene. Avoid altering the verb structure (e.g., do not remove or change the form of the verb).
3.Ensure the Modified Sentence Omits at Least One Key Action or Detail: The modified sentence should omit at least one key action or detail, making it less descriptive than the original. The action or detail omitted should result in a change in the meaning or completeness of the sentence.
4.Avoid simply removing adjectives, adverbs, or other modifying elements: the revised sentence should not be entirely identical to the original in meaning. For instance, When multiple subjects appear in parallel, such as concurrent actions, parallel subjects or objects, one or more may be omitted, but at least one must be retained, resulting in the sentence describing a default state.
5.Where none of the above rules can be applied to modify a sentence, simply return the subject of the original sentence.
- Input Sentence: A man in a pink shirt climbs a rock face.
- Output Sentence: A man in a pink shirt.
- Input Sentence: A boys jumps into the water upside down.
- Output Sentence: A boys jumps into the water.
- Input Sentence: This is a young boy playing with a dollhouse.
- Output Sentence: A young boy.
- Input Sentence: A man wearing a cap and glasses is fixing the seat of a bicycle.
- Output Sentence: A man wearing a cap is fixing the seat of a bicycle.
- Input Sentence:A young boy is frantically staring and shaking his hands.
- Output Sentence: A young boy is frantically staring.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
        prompt2 = (
            "Please process the following sentences in batches according to the rules, outputting one modified sentence per line:\n"
            + "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
        )
        try:
            logging.debug(f"发送API请求，批次大小: {len(text_list)}")
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system", "content": prompt1},
                    {"role": "user", "content": prompt2}
                ],
                temperature=0.3,
                timeout=60  # 设置超时
            )
            # 处理API返回的结果，去除开头的数字和点
            processed_results = []
            for line in completion.choices[0].message.content.strip().split('\n'):
                line = line.strip()
                if line:
                    # 去除开头的 "1. ", "2. " 等编号
                    if '. ' in line and line.split('. ')[0].isdigit():
                        line = line.split('. ', 1)[1]
                    processed_results.append(line)
            logging.debug(f"API响应处理完成，返回 {len(processed_results)} 个结果")
            return processed_results
        except Exception as e:
            logging.error(f"API调用错误: {e}")
            return text_list

    requests_per_minute = 50
    delay_between_requests = 60 / requests_per_minute

    # 批次处理
    batch_size = 20  # 减小批次大小以避免超时
    max_retries = 3
    modified_count = 0

    # 创建文件2列表的副本用于修改
    modified_file2_lines = file2_lines.copy()

    # 进度显示 - 修复变量名错误
    if is_tty:
        main_pbar = tqdm(total=num_to_modify,  # 使用正确的变量名 num_to_modify
                        desc=f"修改相同行", 
                        unit="text",
                        ncols=100)
    else:
        main_pbar = SimpleProgress(num_to_modify, desc=f"修改相同行")  # 使用正确的变量名

    batch_count = 0
    total_batches = (num_to_modify + batch_size - 1) // batch_size
    logging.info(f"批次信息 - 总批次数: {total_batches}, 批次大小: {batch_size}")
    
    # 将需要修改的索引分批处理
    remaining_indices = indices_to_modify.copy()
    
    while modified_count < num_to_modify and remaining_indices:
        # 获取当前批次的索引
        current_batch_size = min(batch_size, len(remaining_indices))
        current_batch_indices = remaining_indices[:current_batch_size]
        remaining_indices = remaining_indices[current_batch_size:]
        
        # 获取对应的文本
        current_batch_texts = [modified_file2_lines[idx] for idx in current_batch_indices]
        
        batch_count += 1
        logging.info(f"开始处理第 {batch_count}/{total_batches} 批次")
        
        retry_count = 0
        failed_indices = current_batch_indices.copy()
        
        while retry_count < max_retries and failed_indices:
            try:
                # 准备当前需要处理的文本
                current_failed_texts = [modified_file2_lines[idx] for idx in failed_indices]
                
                logging.info(f"重试 {retry_count + 1}: 发送 {len(failed_indices)} 个文本到API")
                batch_modified = generate_noisy_text_batch(current_failed_texts)
                
                if len(batch_modified) == len(failed_indices):
                    new_failed_indices = []
                    success_in_batch = 0
                    
                    for j, idx in enumerate(failed_indices):
                        original_text = modified_file2_lines[idx]
                        modified_text = batch_modified[j]
                        
                        if modified_text and modified_text != original_text:
                            # ✅ 成功修改：更新列表中的对应位置
                            modified_file2_lines[idx] = modified_text
                            modified_count += 1
                            success_in_batch += 1
                            
                            # 更新进度条
                            main_pbar.update(1)
                        else:
                            # ❌ 修改失败：保持原始文本，加入重试列表
                            new_failed_indices.append(idx)
                    
                    failed_indices = new_failed_indices
                    logging.info(f"批次处理结果 - 成功: {success_in_batch}, 失败: {len(failed_indices)}")
                    
                    if not failed_indices:
                        logging.info("当前批次所有文本处理成功")
                        break  # 全部成功，退出重试循环
                        
                else:
                    logging.warning(f"API返回数量不匹配: 期望 {len(failed_indices)}, 实际 {len(batch_modified)}")
                    failed_indices = []  # 这种情况下全部重试
                    
            except Exception as e:
                logging.error(f"API调用异常: {e}")
            
            retry_count += 1
            if retry_count < max_retries and failed_indices:
                logging.info(f"准备第 {retry_count} 次重试，剩余 {len(failed_indices)} 个文本")
                time.sleep(5)
        
        # 最终统计
        success_count = len(current_batch_indices) - len(failed_indices)
        if success_count > 0:
            logging.info(f"批次完成 - 成功修改: {success_count} 个文本")
        if failed_indices:
            logging.warning(f"批次完成 - 保持原文本: {len(failed_indices)} 个文本")
        
        # 请求间隔
        if remaining_indices:
            time.sleep(delay_between_requests)
    
    # 关闭进度条
    main_pbar.close()

    logging.info(f"相同行修改完成 - 实际修改: {modified_count}/{num_to_modify}")

    # 保存修改后的文件2列表到文件3
    try:
        with open(file3_path, 'w', encoding='utf-8') as f:
            for text in modified_file2_lines:
                f.write(text + "\n")
        logging.info(f"文件保存成功: {file3_path}")
    except Exception as e:
        logging.error(f"文件保存失败: {e}")
        return

    logging.info(f"文件比对和修改完成")
    logging.info("=" * 70)

# 使用示例
if __name__ == "__main__":
    # 设置日志
    log_file = setup_logging()
    
    # 配置参数
    file1_path = "/home/zbm/xjd/NPC-master/dataset/incomplete_description_noise_5error_MSCOCO/annotations/scan_split/0_noise_train_caps.txt"
    file2_path = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/1.0_noise_train_caps.txt"
    file3_path = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/1.0_noise_train_caps_complete.txt"
    replace_ratio = 1.0
    
    logging.info("=" * 80)
    logging.info("开始文件比对和修改（优化版本）")
    logging.info(f"参考文件: {file1_path}")
    logging.info(f"输入文件: {file2_path}")
    logging.info(f"输出文件: {file3_path}")
    logging.info(f"替换比例: {replace_ratio}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 80)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 调用优化版本的文件比对和修改函数
        compare_and_modify_files_fast(file1_path, file2_path, file3_path, replace_ratio)
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"程序执行完成 - 总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info("文件比对和修改完成！")
        
    except KeyboardInterrupt:
        logging.warning("程序被用户中断")
    except Exception as e:
        logging.error(f"程序执行出错: {e}", exc_info=True)
        raise