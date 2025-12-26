import os
from pathlib import Path

def split_file_with_path(input_file, output_path, lines_per_file=1000):
    """
    更灵活的文件分割函数，支持多种路径格式
    
    参数:
        input_file: 输入文件路径
        output_path: 输出路径（可以是目录或完整文件路径）
        lines_per_file: 每个文件包含的行数
    """
    
    try:
        # 处理输入文件路径
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"错误：输入文件 '{input_file}' 不存在")
            return
        
        # 处理输出路径
        output_path = Path(output_path)
        
        # 如果输出路径是目录
        if output_path.suffix == '':
            # 创建目录
            output_path.mkdir(parents=True, exist_ok=True)
            output_dir = output_path
            output_prefix = "split_file"
        else:
            # 如果输出路径是文件路径
            output_dir = output_path.parent
            output_prefix = output_path.stem
            # 创建目录
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"输入文件: {input_path}")
        print(f"输出目录: {output_dir}")
        print(f"文件前缀: {output_prefix}")
        
        # 使用高效方法分割文件
        file_count = 0
        line_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            while True:
                file_count += 1
                output_file = output_dir / f"{output_prefix}_{file_count:03d}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    for _ in range(lines_per_file):
                        line = f.readline()
                        if not line:
                            break
                        out_f.write(line)
                        line_count += 1
                
                print(f"已创建: {output_file}")
                
                if not line:
                    break
        
        print(f"\n分割完成！")
        print(f"生成文件数量: {file_count}")
        print(f"总行数: {line_count}")
        print(f"保存位置: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 多种使用方式示例：
    
    # 方式1：只指定目录
    # split_file_with_path("your_text_file.txt", "output_folder", 1000)
    
    # 方式2：指定完整路径模式
    # split_file_with_path("your_text_file.txt", "output_folder/data_part", 1000)
    
    # 方式3：使用绝对路径
    split_file_with_path("/home/zbm/xjd/NPC-master/dataset/incomplete_description_noise_5error_MSCOCO/annotations/scan_split/0_noise_train_caps.txt", "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/original", 1000)
    
    # 交互式版本
    # input_file = input("请输入输入文件路径: ").strip()
    # output_path = input("请输入输出路径（目录或文件路径）: ").strip()
    
    try:
        lines = int(input("请输入每个文件的行数（默认1000）: ") or "1000")
    except ValueError:
        lines = 1000
        print("使用默认值: 1000")
    
    # split_file_with_path(input_file, output_path, lines)