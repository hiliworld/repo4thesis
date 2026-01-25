import os
import glob
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler # 【改进】改用 RobustScaler
import warnings

warnings.filterwarnings('ignore')

# === 配置 ===
METRIC_DIR = "data/processed_metrics"
LOG_DIR = "data/processed_logs"
OUTPUT_DIR = "data/final_dataset"
WINDOW_SIZE = 100
TRAIN_RATIO = 0.8
FIXED_FEATURE_DIM = 333 

# 【改进】插值限制：超过3个点缺失就不乱补了
INTERPOLATE_LIMIT = 3
# 【改进】方差阈值调低，防止误删“平时不动、故障才动”的指标（如ErrorCounter）
MIN_VARIANCE = 1e-6 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. 双向对齐器 (Bidirectional Aligner) ===
class DataAlignerV4:
    def __init__(self, window_size=100):
        self.window_size = window_size
        # 日志窗口偏移量：不再是只看未来，而是看前后
        # 假设 Metric 频率是 1秒1个点，Logs 我们看 [T-5s, T+5s]
        # 注意：这里的 window_size 是 Metric 的窗口长度(序列长度)，不是对齐窗口
        # 我们这里简化处理：每个 Metric 点 T，匹配 [T-5, T+5] 范围内的日志
        self.log_search_radius = 5.0 # 秒

    def align(self, metric_df, log_df):
        m_ts = metric_df['timestamp'].values.astype(np.float64)
        
        if log_df is None or log_df.empty:
            return [] 

        l_ts = log_df['timestamp'].values.astype(np.float64)
        l_vals = log_df['event_id'].values

        # 排序日志
        if len(l_ts) > 0:
            sort_idx = np.argsort(l_ts)
            l_ts = l_ts[sort_idx]
            l_vals = l_vals[sort_idx]

        # 【改进】双向窗口搜索
        # 找到 [T - 5, T + 5] 的日志
        left_indices = np.searchsorted(l_ts, m_ts - self.log_search_radius)
        right_indices = np.searchsorted(l_ts, m_ts + self.log_search_radius)

        aligned_samples = []
        
        for i in range(len(m_ts)):
            l_start = left_indices[i]
            l_end = right_indices[i]
            
            if l_end > l_start:
                seq = l_vals[l_start:l_end]
                # 截断过长序列
                if len(seq) > 50: seq = seq[:50]
                
                if len(seq) > 0:
                    aligned_samples.append({
                        'metric_idx': i,
                        'log_seq': seq
                    })
        
        return aligned_samples

# === 2. V4 核心清洗逻辑 ===
def clean_df_v4(df):
    timestamp = df['timestamp']
    feature_cols = [c for c in df.columns if c != 'timestamp']
    metrics = df[feature_cols]

    # A. 【改进】受限线性插值
    # limit=3: 连续缺3个以内才补，缺多了保持 NaN 或填 0，避免伪造趋势
    metrics = metrics.interpolate(method='linear', limit=INTERPOLATE_LIMIT, limit_direction='both').fillna(0)
    
    # B. 【改进】物理意义修正的差分
    for col in metrics.columns:
        # 简单判定累计指标：如果单调增且最大值很大
        if metrics[col].is_monotonic_increasing and metrics[col].max() > 100:
            diff_col = metrics[col].diff().fillna(0)
            # 【关键】差分后小于0的值强制归零（物理意义：计数器重置或数据回滚，不能是负增量）
            diff_col[diff_col < 0] = 0
            metrics[col] = diff_col

    # C. 方差筛选 (保留低方差但可能关键的指标)
    selector = metrics.var() > MIN_VARIANCE
    # 强制保留至少 10 个维度，防止全删光
    if selector.sum() > 10: 
        metrics = metrics.loc[:, selector]

    # D. 【改进】移除 3σ 暴力截断
    # 这里不再做 Clip，把保留异常值的任务交给 RobustScaler

    metrics['timestamp'] = timestamp
    return metrics

def build_final_dataset():
    print("🏭 [Data Manager V4] 开始构建最终数据集 (反泄露+物理修正版)...")
    
    # 清理旧文件
    os.system(f"rm -rf {OUTPUT_DIR}/*.pt")
    
    metric_files = glob.glob(os.path.join(METRIC_DIR, "*.csv"))
    aligner = DataAlignerV4(window_size=WINDOW_SIZE)
    processed_count = 0
    
    for f_metric in tqdm(metric_files, desc="Processing"):
        try:
            metric_filename = os.path.basename(f_metric)
            service_name_core = metric_filename.split('.')[0]
            save_name = metric_filename.replace(".csv", "")
            
            # 1. 读取并做基础清洗 (Diff, Interpolate)
            df_metric = pd.read_csv(f_metric)
            df_metric = clean_df_v4(df_metric) # V4 清洗
            
            raw_vals = df_metric.drop(columns=['timestamp']).values
            curr_dim = raw_vals.shape[1]
            
            # --- Masking ---
            metric_mask = torch.zeros(FIXED_FEATURE_DIM, dtype=torch.float)
            if curr_dim > FIXED_FEATURE_DIM:
                raw_vals = raw_vals[:, :FIXED_FEATURE_DIM]
                metric_mask[:] = 1.0
            else:
                metric_mask[:curr_dim] = 1.0 # 标记真实存在的维度

            # 2. 【核心改进】反数据泄露 Scaling
            # 必须先切分，再 Fit！
            split_point = int(len(raw_vals) * TRAIN_RATIO)
            if split_point < 10: continue

            train_raw = raw_vals[:split_point]
            test_raw = raw_vals[split_point:]

            # 使用 RobustScaler (基于中位数和IQR，不惧怕 35000 这种极值)
            # 它会将 35000 缩放到一个较大的数 (比如 50)，但不会是无穷大，保留了“我是异常”的信息
            scaler = RobustScaler()
            
            # 【关键】只在训练集上 Fit
            scaler.fit(train_raw)
            
            # Transform 全体
            train_norm = scaler.transform(train_raw)
            test_norm = scaler.transform(test_raw)
            
            # 合并回 huge array 方便后续 Padding
            norm_vals = np.vstack([train_norm, test_norm])
            
            # 【改进】软截断 (Soft Clipping)
            # 虽然 RobustScaler 能处理极值，但为了神经网络稳定性，我们做一个宽范围截断
            # ±10 对于 RobustScaler 来说已经是非常极端的异常了 (通常 99% 的数据在 ±3 以内)
            norm_vals = np.clip(norm_vals, -10, 10)

            # --- Padding ---
            if curr_dim < FIXED_FEATURE_DIM:
                pad_width = FIXED_FEATURE_DIM - curr_dim
                norm_vals = np.pad(norm_vals, ((0,0), (0, pad_width)), 'constant', constant_values=0)
            
            metric_tensor = torch.tensor(norm_vals, dtype=torch.float32)
            timestamps = df_metric['timestamp'].values
            
            # 3. 对齐日志
            f_log = os.path.join(LOG_DIR, f"{service_name_core}.csv")
            if not os.path.exists(f_log): continue
            df_log = pd.read_csv(f_log)
            
            alignment_info = aligner.align(df_metric, df_log)
            if len(alignment_info) < 10: continue

            # 4. 提取与保存
            valid_indices = [x['metric_idx'] for x in alignment_info]
            valid_logs = [x['log_seq'] for x in alignment_info]
            
            filtered_metrics = metric_tensor[valid_indices]
            filtered_timestamps = timestamps[valid_indices]
            
            # 重新计算切分点 (因为对齐后样本数变了)
            final_num_samples = len(filtered_metrics)
            final_split_idx = int(final_num_samples * TRAIN_RATIO)
            
            # 保存元数据 (Metadata) - 方便溯源
            metadata = {
                'service': service_name_core,
                'scaler_center': scaler.center_, # 均值/中位数
                'scaler_scale': scaler.scale_,   # 标准差/IQR
                'n_features': curr_dim
            }
            joblib.dump(metadata, os.path.join(OUTPUT_DIR, f"{save_name}_meta.pkl"))

            torch.save({
                'metrics': filtered_metrics[:final_split_idx], 
                'logs': valid_logs[:final_split_idx],           
                'timestamps': filtered_timestamps[:final_split_idx],
                'metric_mask': metric_mask
            }, os.path.join(OUTPUT_DIR, f"{save_name}_train.pt"))
            
            torch.save({
                'metrics': filtered_metrics[final_split_idx:],
                'logs': valid_logs[final_split_idx:],
                'timestamps': filtered_timestamps[final_split_idx:],
                'metric_mask': metric_mask
            }, os.path.join(OUTPUT_DIR, f"{save_name}_test.pt"))
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ {metric_filename}: {e}")

    print(f"🎉 V4 数据处理完成！成功: {processed_count}")

if __name__ == "__main__":
    build_final_dataset()