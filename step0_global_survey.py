import os
import pandas as pd
from tqdm import tqdm

# === 配置 ===
# 你的数据集根目录
TARGET_DIR = "data/Aiops-Dataset/data"
# 结果保存到哪里
REPORT_FILE = "dataset_structure_report.txt"
# 每个文件夹下只看前 N 个文件 (防止刷屏，如果你想看全部，设为 None)
SAMPLE_PER_DIR = 3 

def survey_files():
    print(f"🚀 开始全域普查: {TARGET_DIR}")
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f_out:
        f_out.write(f"=== AIOps Dataset 全域文件结构普查 ===\n")
        f_out.write(f"目标路径: {TARGET_DIR}\n\n")

        # os.walk 是遍历文件夹的神器
        for root, dirs, files in os.walk(TARGET_DIR):
            # 过滤出 csv 文件
            csv_files = [x for x in files if x.endswith('.csv')]
            
            if not csv_files:
                continue
                
            # 记录当前文件夹路径
            f_out.write(f"\n{'='*50}\n")
            f_out.write(f"📂 目录: {root}\n")
            f_out.write(f"   包含 CSV 文件数: {len(csv_files)}\n")
            f_out.write(f"{'='*50}\n")
            
            # 采样检查
            check_list = csv_files[:SAMPLE_PER_DIR] if SAMPLE_PER_DIR else csv_files
            
            for fname in check_list:
                full_path = os.path.join(root, fname)
                f_out.write(f"\n📄 文件: {fname}\n")
                
                try:
                    # 只读取前 5 行 (纯文本模式，不依赖 Pandas 解析，更稳健)
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        head_lines = [next(f_in).strip() for _ in range(5)]
                    
                    # 写入报告
                    for idx, line in enumerate(head_lines):
                        f_out.write(f"   [L{idx}] {line}\n")
                        
                except Exception as e:
                    f_out.write(f"   ❌ 读取失败: {e}\n")

    print(f"✅ 普查完成！请查看报告文件: {REPORT_FILE}")
    print(f"   (建议使用 'less {REPORT_FILE}' 命令查看)")

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误: 找不到目录 {TARGET_DIR}")
    else:
        survey_files()