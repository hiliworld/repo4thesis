import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# 忽略 pandas 的 FutureWarning (会让输出很乱)
warnings.simplefilter(action='ignore', category=FutureWarning)

# === 配置 ===
DATA_ROOT = "data/Aiops-Dataset/data"
OUTPUT_DIR = "data/processed_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_metrics():
    print(f"🏭 [Metric Factory] 启动！扫描路径: {DATA_ROOT}")
    
    # 我们用一个字典在内存里暂存数据: 
    # data_buffer[cmdb_id] = [DataFrame1, DataFrame2, ...]
    # 最后再合并。这样比反复读写磁盘快。
    data_buffer = {}
    
    # 1. 递归扫描所有 metric 目录下的 csv
    # 逻辑: data/日期/metric/类别/*.csv
    search_pattern = os.path.join(DATA_ROOT, "*", "metric", "*", "*.csv")
    all_files = glob.glob(search_pattern)
    print(f"   -> 发现 {len(all_files)} 个指标文件")
    
    for file_path in tqdm(all_files, desc="Parsing Metrics"):
        try:
            # 2. 读取 CSV
            # 这里的列名参考了你的报告：timestamp, cmdb_id, kpi_name, value
            # 注意：metric_service.csv 格式特殊，需要单独处理
            filename = os.path.basename(file_path)
            
            if "metric_service.csv" in filename:
                # 特殊处理: service, timestamp, rr, sr, mrt, count
                df = pd.read_csv(file_path)
                # 把 rr, sr, mrt, count 变成 Long Format 方便统一处理
                # id_vars 保留列，value_vars 要转换的列
                df = df.melt(id_vars=['timestamp', 'service'], 
                             value_vars=['rr', 'sr', 'mrt', 'count'],
                             var_name='kpi_name', value_name='value')
                df.rename(columns={'service': 'cmdb_id'}, inplace=True)
            else:
                # 标准 Long Format
                df = pd.read_csv(file_path)
            
            # 3. 按 cmdb_id 分组放入缓冲区
            # 因为一个文件里可能包含多个机器的数据
            for cmdb_id, group in df.groupby('cmdb_id'):
                # 只保留需要的列
                clean_group = group[['timestamp', 'kpi_name', 'value']].copy()
                
                # 转换时间戳为 Int (秒级)
                # 你的报告里是 1651507200 (秒) 或 1651507200154 (毫秒)
                # 统一转为秒 (如果是 13 位数字，除以 1000)
                if clean_group['timestamp'].iloc[0] > 1e11: 
                    clean_group['timestamp'] = clean_group['timestamp'] // 1000
                
                if cmdb_id not in data_buffer:
                    data_buffer[cmdb_id] = []
                data_buffer[cmdb_id].append(clean_group)
                
        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}: {e}")

    # 4. 合并 & 透视 (Pivot)
    print(f"🧩 正在合并 {len(data_buffer)} 个实体的指标...")
    for cmdb_id, df_list in tqdm(data_buffer.items(), desc="Merging & Pivoting"):
        try:
            # A. 垂直合并 (Concat)
            full_df = pd.concat(df_list, ignore_index=True)
            
            # B. 透视 (Pivot) -> 变成宽表
            # index=Time, columns=KPI, values=Value
            # 如果同一秒有重复 KPI (比如双重采集)，取 mean
            pivot_df = full_df.pivot_table(
                index='timestamp', 
                columns='kpi_name', 
                values='value', 
                aggfunc='mean'
            )
            
            # C. 填充缺失值
            # 前向填充 (ffill) + 填 0
            pivot_df = pivot_df.fillna(method='ffill').fillna(0)
            
            # D. 保存
            # 处理文件名中的非法字符 (比如 / 或 :)
            safe_name = cmdb_id.replace("/", "_").replace(":", "_")
            save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
            pivot_df.to_csv(save_path)
            
        except Exception as e:
            print(f"❌ 处理实体 {cmdb_id} 失败: {e}")

    print(f"✅ 指标工厂完工！产出位于: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_metrics()