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

    def align(self, metric_df, log_df, metric_ts_col='timestamp', log_ts_col='timestamp', log_val_col='EventId'):
        """
        参数:
        metric_df: 数值指标 DataFrame
        log_df: 日志 DataFrame (必须包含 EventId, 也就是说需要先经过 Drain3 解析)
        """
        print(f"⚡ [Aligner] 开始对齐... Window={self.window_size}s")
        
        # 1. 预处理：确保时间戳是 Float/Int 类型且已排序
        # (如果是字符串，需要先 pd.to_datetime 再 .timestamp())
        m_ts = metric_df[metric_ts_col].values.astype(np.float64)
        l_ts = log_df[log_ts_col].values.astype(np.float64)
        l_vals = log_df[log_val_col].values # 日志的内容 (EventId)

        # 确保 Log 是按时间排序的 (SearchSorted 的前提)
        sort_idx = np.argsort(l_ts)
        l_ts = l_ts[sort_idx]
        l_vals = l_vals[sort_idx]

        # 2. 核心魔法：使用 searchsorted 快速定位
        # 对于每个 metric 时间点 T，我们需要找到 log 时间轴上的 [T, T+window) 区间
        # left_indices[i] 意味着：在 Log 时间轴上，第一个 >= m_ts[i] 的位置在哪？
        print("   -> 正在计算区间索引 (Binary Search)...")
        left_indices = np.searchsorted(l_ts, m_ts)
        right_indices = np.searchsorted(l_ts, m_ts + self.window_size)

        # 3. 组装数据
        aligned_samples = []
        
        # 这一步不可避免需要循环，但因为索引已经算好了，所以只是简单的切片操作
        # 我们只遍历 Metric，因为它是基准
        print("   -> 正在切片组装...")
        for i in range(len(m_ts)):
            # 获取当前 Metric 窗口对应的 Log 范围
            l_start = left_indices[i]
            l_end = right_indices[i]
            
            # 切片取出这段时间的日志
            if l_end > l_start:
                # Case 1: 有日志
                logs_in_window = l_vals[l_start:l_end]
                # 这里可以做一个截断，防止某个窗口日志太多撑爆内存
                if len(logs_in_window) > 50: 
                    logs_in_window = logs_in_window[:50]
            else:
                # Case 2: 没日志 -> 填 0 (Padding)
                # 注意：我们约定 0 是 PAD，所以正常的 EventId 应该从 1 开始
                logs_in_window = np.array([0]) 

            # 构造样本
            # 这里我们先把 Metric 的当前时刻值拿出来，或者拿整个窗口的 Metric
            # 简单起见，我们先拿 metrics 的行索引，后面 Dataset 再去查具体数值
            aligned_samples.append({
                'metric_idx': i,             # 指向 metric_df 的第几行
                'timestamp': m_ts[i],
                'log_sequence': logs_in_window # 这是一个变长的 numpy array
            })

        print(f"✅ 对齐完成！生成样本数: {len(aligned_samples)}")
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