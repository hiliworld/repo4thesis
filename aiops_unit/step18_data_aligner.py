import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

class DataAligner:
    """
    多模态数据对齐器 (High-Performance)
    目标：将 '连续的 Metric 流' 与 '离散的 Log 流' 在时间维度上强行对齐。
    """
    def __init__(self, window_size_sec=60):
        self.window_size = window_size_sec

    def align(self, metric_df, log_df):
        m_ts = metric_df['timestamp'].values.astype(np.float64)
        
        # 处理空日志情况
        if log_df is None or log_df.empty:
            # 【修复 1】如果没有日志，直接返回空列表
            # 这种服务根本没法训练对比学习，直接丢弃
            return []

        l_ts = log_df['timestamp'].values.astype(np.float64)
        l_vals = log_df['event_id'].values

        # 排序 (保持原样)
        if len(l_ts) > 0:
            sort_idx = np.argsort(l_ts)
            l_ts = l_ts[sort_idx]
            l_vals = l_vals[sort_idx]

        # 快速对齐 (保持原样)
        left_indices = np.searchsorted(l_ts, m_ts)
        right_indices = np.searchsorted(l_ts, m_ts + self.window_size)

        aligned_samples = []
        
        # 【修复 2】统计有效样本率
        valid_count = 0
        total_count = len(m_ts)
        
        for i in range(total_count):
            l_start = left_indices[i]
            l_end = right_indices[i]
            
            # 【核心修复逻辑】
            # 只有当窗口内真的有日志时，才算有效样本
            # 我们甚至可以要求：至少有 2 条日志才算有效 (避免偶然噪声)
            if l_end > l_start: 
                seq = l_vals[l_start:l_end]
                # 截断
                if len(seq) > 50: 
                    seq = seq[:50]
                
                # 【关键】只有非空序列才加入
                if len(seq) > 0:
                    aligned_samples.append({
                        'metric_idx': i, # 记录对应的 metric 索引，一会切片用
                        'log_seq': seq
                    })
                    valid_count += 1
            
            # 注意：这里不再 append 那些只有 [0] 的样本了！
        
        print(f"   -> 对齐统计: 总窗口 {total_count}, 有效含日志窗口 {valid_count} ({(valid_count/total_count)*100:.2f}%)")
        return aligned_samples
    
# === 单元测试 (Unit Test) ===
# 每次写完核心算法，必须立刻自己造数据测一下，不要等跑全量才发现 Bug
if __name__ == "__main__":
    # 1. 造点假指标 (0, 10, 20... 90 秒)
    df_metric = pd.DataFrame({
        'timestamp': np.arange(0, 100, 10),
        'cpu': np.random.rand(10)
    })
    
    # 2. 造点假日志 (5秒, 12秒, 15秒...)
    # EventId 这里的 0 是保留位，我们用 1, 2, 3
    df_log = pd.DataFrame({
        'timestamp': [5, 12, 15, 85, 200], 
        'EventId':   [1,  2,  3,  4,   5]
    })
    
    # 3. 运行对齐 (窗口 20秒)
    aligner = DataAligner(window_size_sec=20)
    results = aligner.align(df_metric, df_log)
    
    # 4. 打印结果验证
    print("\n=== 验证结果 ===")
    for res in results[:5]: # 只看前5个
        ts = res['timestamp']
        logs = res['log_sequence']
        print(f"Time {ts} ~ {ts+20}: Logs -> {logs}")
        
    # 预期解析：
    # Time 0~20: 应该包含 timestamp 5, 12, 15 -> Logs [1, 2, 3]
    # Time 10~30: 应该包含 timestamp 12, 15 -> Logs [2, 3]
    # Time 20~40: 空 -> Logs [0]