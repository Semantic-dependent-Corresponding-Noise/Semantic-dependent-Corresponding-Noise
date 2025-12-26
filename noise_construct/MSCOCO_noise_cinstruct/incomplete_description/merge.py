import os
import glob
import re
from tqdm import tqdm

def merge_noisy_files(input_dir, output_file, start_num=1, end_num=None):
    """
    将 noisy_split_file_*.txt 文件按顺序合并成一个大文件
    
    参数:
        input_dir: 输入文件目录
        output_file: 输出文件路径
        start_num: 起始文件编号
        end_num: 结束文件编号（None表示自动检测）
    """
    
    print("🔄 开始合并噪声文本文件")
    print(f"📁 输入目录: {input_dir}")
    print(f"💾 输出文件: {output_file}")
    
    # 自动检测文件范围
    if end_num is None:
        existing_files = glob.glob(os.path.join(input_dir, "noisy_split_file_*.txt"))
        if existing_files:
            file_numbers = []
            for f in existing_files:
                try:
                    # 从文件名中提取数字
                    num = int(re.findall(r'\d+', os.path.basename(f))[0])
                    file_numbers.append(num)
                except (IndexError, ValueError):
                    continue
            if file_numbers:
                end_num = max(file_numbers)
                print(f"🔍 自动检测到文件范围: {start_num:03d} 到 {end_num:03d}")
            else:
                print("❌ 无法从文件名中提取数字")
                return
        else:
            print("❌ 没有找到 noisy_split_file_*.txt 文件")
            return
    
    # 生成要处理的文件列表
    file_paths = []
    missing_files = []
    
    for file_num in range(start_num, end_num + 1):
        filename = f"noisy_split_file_{file_num:03d}.txt"
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            missing_files.append(filename)
    
    if not file_paths:
        print("❌ 没有找到可用的文件")
        return
    
    print(f"📊 找到 {len(file_paths)} 个文件")
    if missing_files:
        print(f"⚠️  缺失 {len(missing_files)} 个文件: {', '.join(missing_files[:5])}{'...' if len(missing_files) > 5 else ''}")
    
    # 显示文件列表
    print("\n📋 文件处理顺序:")
    for i, file_path in enumerate(file_paths[:10]):
        print(f"  {i+1:2d}. {os.path.basename(file_path)}")
    if len(file_paths) > 10:
        print(f"  ... 还有 {len(file_paths) - 10} 个文件")
    
    # 统计总行数
    print("\n🔢 正在统计总行数...")
    total_lines = 0
    file_info = []
    
    for file_path in tqdm(file_paths, desc="统计行数"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_lines += line_count
                file_info.append({
                    'path': file_path,
                    'lines': line_count,
                    'content': lines
                })
        except Exception as e:
            print(f"❌ 读取文件失败 {os.path.basename(file_path)}: {e}")
            file_info.append({
                'path': file_path,
                'lines': 0,
                'content': []
            })
    
    print(f"📈 总行数: {total_lines:,}")
    
    # 合并文件
    print("\n🔄 开始合并文件...")
    successful_files = 0
    failed_files = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with tqdm(total=total_lines, desc="合并进度", unit="line", ncols=80) as pbar:
                for info in file_info:
                    filename = os.path.basename(info['path'])
                    
                    if info['lines'] > 0:
                        try:
                            # 写入文件内容
                            for line in info['content']:
                                out_f.write(line)
                            
                            successful_files += 1
                            pbar.update(info['lines'])
                            pbar.set_description(f"合并: {filename}")
                            
                        except Exception as e:
                            print(f"❌ 写入文件失败 {filename}: {e}")
                            failed_files += 1
                            pbar.update(info['lines'])
                    else:
                        print(f"⚠️  跳过空文件: {filename}")
                        failed_files += 1
                        pbar.update(0)
    
    except Exception as e:
        print(f"❌ 创建输出文件失败: {e}")
        return
    
    # 验证输出文件
    print("\n🔍 验证输出文件...")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_lines = sum(1 for _ in f)
        print(f"✅ 输出文件行数: {output_lines:,}")
        
        if output_lines == total_lines:
            print("🎉 文件行数验证通过！")
        else:
            print(f"⚠️  行数不匹配: 期望 {total_lines}, 实际 {output_lines}")
            
    except Exception as e:
        print(f"❌ 验证输出文件失败: {e}")
    
    # 输出统计信息
    print("\n📊 合并完成统计:")
    print(f"✅ 成功合并: {successful_files} 个文件")
    print(f"❌ 失败/跳过: {failed_files} 个文件")
    print(f"📄 总文件数: {len(file_paths)} 个")
    print(f"📝 总文本行数: {total_lines:,} 行")
    print(f"💾 输出文件: {output_file}")

def find_missing_files(input_dir, start_num=1, end_num=None):
    """
    查找缺失的文件
    """
    if end_num is None:
        existing_files = glob.glob(os.path.join(input_dir, "noisy_split_file_*.txt"))
        if existing_files:
            file_numbers = [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in existing_files]
            end_num = max(file_numbers)
        else:
            print("没有找到文件")
            return
    
    missing_files = []
    for file_num in range(start_num, end_num + 1):
        filename = f"noisy_split_file_{file_num:03d}.txt"
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        print(f"缺失 {len(missing_files)} 个文件:")
        for i, filename in enumerate(missing_files):
            print(f"  {i+1:2d}. {filename}")
    else:
        print("✅ 所有文件都存在")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_directory = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/noise"
    output_file_path = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/1.0_noise_train_caps.txt"
    
    # 可选：检查缺失文件
    # print("🔍 检查缺失文件...")
    # find_missing_files(input_directory, start_num=1, end_num=567)
    
    # 合并文件
    merge_noisy_files(
        input_dir=input_directory,
        output_file=output_file_path,
        start_num=1,
        end_num=None  # 自动检测
    )