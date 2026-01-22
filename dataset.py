import torch
from torch.utils.data import Dataset
import os
import glob
from tqdm import tqdm
import numpy as np

class UACDataset(Dataset):
    def __init__(self, data_dir, mode='train', metric_window=20, log_max_len=50):
        """
        Args:
            data_dir (str): Step 19 产出的 data/final_dataset 目录
            mode (str): 'train' 或 'test'
            metric_window (int): 指标序列的回看窗口长度
            log_max_len (int): 日志序列的最大长度
        """
        self.metric_window = metric_window
        self.log_max_len = log_max_len
        self.mode = mode
        
        # 1. 扫描文件
        pattern = os.path.join(data_dir, f"*_{mode}.pt")
        files = glob.glob(pattern)
        
        if len(files) == 0:
            raise ValueError(f"❌ 在 {data_dir} 下没找到 _{mode}.pt 文件，请检查 Step 19 是否运行成功！")

        print(f"📥 [Dataset] 正在加载 {len(files)} 个 {mode} 文件...")
        
        # 2. 预加载所有数据 (构建全局索引)
        self.global_samples = []
        
        for f in tqdm(files, desc="Loading PT files"):
            try:
                # 显式关闭 weights_only，允许加载 numpy 数组
                content = torch.load(f, weights_only=False)
                
                metrics = content['metrics'] # [Total_Time, 333]
                logs = content['logs']       # List of arrays (Step 19 已经对齐好了)
                timestamps = content['timestamps']
                
                # A. 读取显式掩码 (Explicit Mask) [结合论文: 异构数据处理]
                if 'metric_mask' in content:
                    feature_mask = content['metric_mask']
                else:
                    # 兼容旧数据，但不推荐
                    feature_mask = torch.ones(333, dtype=torch.float)
                
                # B. 构建滑窗样本
                num_steps = metrics.shape[0]
                if num_steps <= metric_window:
                    continue 
                
                for i in range(metric_window, num_steps):
                    # 【针对 Loss 不降的二重保险】
                    # 检查当前时刻的日志是否为空 (防止 Step 19 漏网之鱼)
                    # 如果日志是空的 (len=0) 或者全是 0 (Pad)，跳过！
                    curr_log = logs[i]
                    if len(curr_log) == 0:
                        continue
                    if len(curr_log) == 1 and curr_log[0] == 0:
                        continue

                    self.global_samples.append({
                        'metric_data': metrics,      # 引用 (不占额外内存)
                        'log_data': logs,            # 引用
                        'feature_mask': feature_mask,# 引用
                        'idx': i                     # 当前时刻索引
                    })
                    
            except Exception as e:
                print(f"⚠️ 加载 {f} 失败: {e}")
                
        print(f"✅ {mode} 集加载完毕！有效样本数: {len(self.global_samples)}")

    def __len__(self):
        return len(self.global_samples)

    def __getitem__(self, index):
        # 1. 获取正样本 (Positive Pair)
        sample_info = self.global_samples[index]
        curr_idx = sample_info['idx']
        
        # A. Metric (Anchor)
        # 取 [t-window, t]
        metric_seq = sample_info['metric_data'][curr_idx - self.metric_window : curr_idx]
        feature_mask = sample_info['feature_mask']
        
        # B. Log (Positive)
        log_seq_raw = sample_info['log_data'][curr_idx]
        
        # Log Padding 处理 [结合论文: semantics.py]
        log_seq = torch.zeros(self.log_max_len, dtype=torch.long)
        log_mask = torch.zeros(self.log_max_len, dtype=torch.float)
        
        current_len = len(log_seq_raw)
        if current_len > 0:
            if current_len > self.log_max_len:
                current_len = self.log_max_len
                log_seq_raw = log_seq_raw[:current_len]
            
            log_seq[:current_len] = torch.from_numpy(log_seq_raw)
            log_mask[:current_len] = 1.0 # 真实日志部分为 1

        # === 2. 构造负样本 (Hard Negative Sampling) [结合论文: construct_unmatched_data] ===
        # 目的：打破 Loss 停滞。我们随机找一个“不匹配”的 Metric 给模型，告诉它“这也是错的”。
        # 简单策略：随机采样一个 index != current_index
        if self.mode == 'train':
            neg_idx = np.random.randint(0, len(self.global_samples))
            while neg_idx == index: # 确保不是自己
                neg_idx = np.random.randint(0, len(self.global_samples))
            
            neg_info = self.global_samples[neg_idx]
            neg_curr_idx = neg_info['idx']
            # 取出负样本 Metric
            neg_metric_seq = neg_info['metric_data'][neg_curr_idx - self.metric_window : neg_curr_idx]
            # 注意：负样本也要带上它自己的 mask (因为不同服务的维度可能不同)
            neg_metric_mask = neg_info['feature_mask']
        else:
            # 测试模式下不需要负样本，给个占位符即可
            neg_metric_seq = metric_seq.clone()
            neg_metric_mask = feature_mask.clone()

        return {
            'metric_seq': metric_seq,       # [Window, 333]
            'metric_mask': feature_mask,    # [333]
            'log_seq': log_seq,             # [50]
            'log_mask': log_mask,           # [50]
            'neg_metric_seq': neg_metric_seq, # [Window, 333] (新增：负样本)
            'neg_metric_mask': neg_metric_mask # [333] (新增：负样本Mask)
        }