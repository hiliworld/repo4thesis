import os
import glob
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# 引入之前的对齐器 (为了方便，这里内嵌简化版逻辑，确保独立运行不报错)
class DataAligner:
    def __init__(self, window_size=100):
        self.window_size = window_size

    def align(self, metric_df, log_df):
        m_ts = metric_df['timestamp'].values.astype(np.float64)
        
        # 处理空日志情况
        if log_df is None or log_df.empty:
            l_ts = np.array([])
            l_vals = np.array([])
        else:
            l_ts = log_df['timestamp'].values.astype(np.float64)
            l_vals = log_df['event_id'].values

        # 排序
        if len(l_ts) > 0:
            sort_idx = np.argsort(l_ts)
            l_ts = l_ts[sort_idx]
            l_vals = l_vals[sort_idx]

        # 快速对齐
        left_indices = np.searchsorted(l_ts, m_ts)
        right_indices = np.searchsorted(l_ts, m_ts + self.window_size)

        aligned_samples = []
        for i in range(len(m_ts)):
            l_start = left_indices[i]
            l_end = right_indices[i]
            
            if l_end > l_start:
                seq = l_vals[l_start:l_end]
                if len(seq) > 50: seq = seq[:50] # 截断
            else:
                seq = np.array([0]) # Padding

            aligned_samples.append({'log_seq': seq})
        
        return aligned_samples

# === 配置 ===
METRIC_DIR = "data/processed_metrics"
LOG_DIR = "data/processed_logs"
OUTPUT_DIR = "data/final_dataset"
WINDOW_SIZE = 100
TRAIN_RATIO = 0.8
# ⚠️ 关键修改：设定一个足够大的固定维度，覆盖所有类型的监控数据
# 你的报错里最大看到了 64，为了保险，我们设为 100 (多余的填0)
FIXED_FEATURE_DIM = 333 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_final_dataset():
    print("🏭 [Step 19 Fix] 开始构建最终数据集 (Per-Service Scaler + Padding)...")
    
    metric_files = glob.glob(os.path.join(METRIC_DIR, "*.csv"))
    print(f"   -> 找到 {len(metric_files)} 个 Metric 文件")
    
    # 只需要实例化对齐器
    aligner = DataAligner(window_size=WINDOW_SIZE)
    
    # 统计信息
    processed_count = 0
    
    for f_metric in tqdm(metric_files, desc="Processing"):
        try:
            service_name = os.path.basename(f_metric).replace(".csv", "")
            
            # 1. 读取 Metric
            df_metric = pd.read_csv(f_metric)
            # 拿到原始数值 (去掉 timestamp)
            raw_vals = df_metric.drop(columns=['timestamp']).values
            
            # --- 关键修改 A: 维度检查与补齐 ---
            # 如果当前维度 > FIXED_FEATURE_DIM，说明我们设小了，报错提示
            curr_dim = raw_vals.shape[1]
            if curr_dim > FIXED_FEATURE_DIM:
                print(f"⚠️ 警告: 文件 {service_name} 维度 ({curr_dim}) 超过了设定值 ({FIXED_FEATURE_DIM})，将被截断！")
                raw_vals = raw_vals[:, :FIXED_FEATURE_DIM]
            
            # --- 关键修改 B: 独立归一化 (Local Scaling) ---
            # 我们不再用全局 Scaler，而是每个服务用自己的 Scaler
            # 这样避免了 "CPU" 和 "Heap Memory" 混在一起算的问题
            split_idx = int(len(raw_vals) * TRAIN_RATIO)
            train_part = raw_vals[:split_idx]
            
            if len(train_part) < 10: # 数据太少，跳过
                continue
                
            scaler = RobustScaler()
            scaler.fit(train_part) # 只在训练集上 fit
            
            norm_vals = scaler.transform(raw_vals) # 转换全部
            
            # --- 关键修改 C: 零填充 (Zero Padding) ---
            # 目标: [T, FIXED_FEATURE_DIM]
            # 当前: [T, curr_dim]
            # 我们需要在右侧补 (FIXED_FEATURE_DIM - curr_dim) 列的 0
            if curr_dim < FIXED_FEATURE_DIM:
                pad_width = FIXED_FEATURE_DIM - curr_dim
                # np.pad(array, ((top, bottom), (left, right)))
                norm_vals = np.pad(norm_vals, ((0,0), (0, pad_width)), 'constant', constant_values=0)
            
            # 转 Tensor
            metric_tensor = torch.tensor(norm_vals, dtype=torch.float32)
            timestamps = df_metric['timestamp'].values
            
            # 2. 寻找 Log 并对齐
            f_log = os.path.join(LOG_DIR, f"{service_name}.csv")
            if os.path.exists(f_log):
                df_log = pd.read_csv(f_log)
            else:
                df_log = None
                
            alignment_info = aligner.align(df_metric, df_log)
            
            # 3. 划分并保存
            # 训练集
            train_data = {
                'metrics': metric_tensor[:split_idx], 
                'logs': [x['log_seq'] for x in alignment_info[:split_idx]], 
                'timestamps': timestamps[:split_idx]
            }
            # 测试集
            test_data = {
                'metrics': metric_tensor[split_idx:],
                'logs': [x['log_seq'] for x in alignment_info[split_idx:]],
                'timestamps': timestamps[split_idx:]
            }
            
            # 保存 .pt
            torch.save(train_data, os.path.join(OUTPUT_DIR, f"{service_name}_train.pt"))
            torch.save(test_data, os.path.join(OUTPUT_DIR, f"{service_name}_test.pt"))
            
            # 我们顺便保存一下这个服务的 scaler，万一以后推理要用
            joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{service_name}_scaler.pkl"))
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理出错 {service_name}: {e}")

    print(f"🎉 处理完成！成功生成 {processed_count} 个服务的对齐数据。")
    print(f"   所有 Metric 维度已统一为: {FIXED_FEATURE_DIM} (不足补0)")

if __name__ == "__main__":
    build_final_dataset()