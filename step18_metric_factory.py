import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import gc

# 忽略 pandas 的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# === 配置 ===
DATA_ROOT = "data/Aiops-Dataset/data"
OUTPUT_DIR = "data/processed_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【关键修改 1】时间桶大小
# 你刚刚确认了原始数据是 60s 间隔，所以这里必须设为 60
# 这样 10:00:01 和 10:00:02 都会被归位到 10:00:00，防止数据稀疏
BUCKET_SIZE = 60 

def process_metrics():
    print(f"🏭 [Metric Factory V2] 启动！扫描路径: {DATA_ROOT}")
    print(f"   -> 时间对齐模式: 强制 {BUCKET_SIZE}s 对齐 (Snap-to-Grid)")
    
    data_buffer = {}
    
    search_pattern = os.path.join(DATA_ROOT, "*", "metric", "*", "*.csv")
    all_files = glob.glob(search_pattern)
    print(f"   -> 发现 {len(all_files)} 个指标文件")
    
    for file_path in tqdm(all_files, desc="Parsing Metrics"):
        try:
            filename = os.path.basename(file_path)
            
            # 1. 读取 CSV
            if "metric_service.csv" in filename:
                df = pd.read_csv(file_path)
                df = df.melt(id_vars=['timestamp', 'service'], 
                             value_vars=['rr', 'sr', 'mrt', 'count'],
                             var_name='kpi_name', value_name='value')
                df.rename(columns={'service': 'cmdb_id'}, inplace=True)
            else:
                df = pd.read_csv(file_path)
            
            # 2. 分组处理
            for cmdb_id, group in df.groupby('cmdb_id'):
                clean_group = group[['timestamp', 'kpi_name', 'value']].copy()
                
                # A. 毫秒转秒
                if clean_group['timestamp'].iloc[0] > 1e11: 
                    clean_group['timestamp'] = clean_group['timestamp'] // 1000
                
                # 【关键修改 2】时间对齐 (Snap to Grid)
                # 这一步会把 1651334402 强行掰成 1651334400
                clean_group['timestamp'] = (clean_group['timestamp'] // BUCKET_SIZE) * BUCKET_SIZE
                
                if cmdb_id not in data_buffer:
                    data_buffer[cmdb_id] = []
                data_buffer[cmdb_id].append(clean_group)
            
            # 及时释放内存
            del df
                
        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}: {e}")

    # 3. 合并 & 透视
    print(f"🧩 正在合并 {len(data_buffer)} 个实体的指标...")
    
    for cmdb_id, df_list in tqdm(data_buffer.items(), desc="Merging & Pivoting"):
        try:
            full_df = pd.concat(df_list, ignore_index=True)
            
            # B. 透视 (Pivot)
            # 如果同一个60s窗口内有多条数据(比如01秒和02秒都有)，取mean合并
            pivot_df = full_df.pivot_table(
                index='timestamp', 
                columns='kpi_name', 
                values='value', 
                aggfunc='mean'
            )
            
            # 【关键修改 3】受限填充 (Safe Fill)
            # 之前的无限制 ffill 会制造大量虚假直线
            # 现在：只允许向前补 1 个点 (即允许 60s 的延迟)，再多就视为断连(0)
            pivot_df = pivot_df.ffill(limit=1).fillna(0)
            
            # C. 保存
            safe_name = cmdb_id.replace("/", "_").replace(":", "_")
            save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
            pivot_df.to_csv(save_path)
            
        except Exception as e:
            print(f"❌ 处理实体 {cmdb_id} 失败: {e}")
        finally:
            del df_list
            
    gc.collect()
    print(f"✅ 指标工厂完工！产出位于: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_metrics()