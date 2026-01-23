import torch
from torch.utils.data import Dataset
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

class UACDataset(Dataset):
    def __init__(self, data_dir, mode='train', metric_window=20, log_max_len=50, vocab_size=3000, 
                 gt_dir='data/Aiops-Dataset/groundtruth'): # 【新增】需要 GT 路径来剔除脏数据
        
        self.metric_window = metric_window
        self.log_max_len = log_max_len
        self.vocab_size = vocab_size 
        self.mode = mode
        
        # === 0. 加载故障时间表 (Blacklist) ===
        # 只有在训练模式下，我们才需要剔除故障数据
        self.fault_ranges = {} # {'service_name': [(start, end), ...]}
        if mode == 'train':
            print(f"🧹 [Dataset] 正在构建故障黑名单，准备清洗训练集...")
            gt_files = glob.glob(os.path.join(gt_dir, "groundtruth-*.csv"))
            for f in gt_files:
                try:
                    df = pd.read_csv(f)
                    for _, row in df.iterrows():
                        svc = str(row['cmdb_id'])
                        ts = int(row['timestamp'])
                        # 标记故障发生的前后窗口为“脏数据”
                        # 比如故障发生后 5 分钟内，或者前 1 分钟(前兆)
                        if svc not in self.fault_ranges:
                            self.fault_ranges[svc] = []
                        # 扩宽一点，宁可错杀一千正常，不可放过一个故障进训练集
                        self.fault_ranges[svc].append((ts - 60, ts + 60)) 
                except: pass
            print(f"   -> 已加载 {len(self.fault_ranges)} 个服务的故障记录")

        # 1. 扫描文件
        pattern = os.path.join(data_dir, f"*_{mode}.pt")
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError(f"❌ 在 {data_dir} 下没找到 _{mode}.pt 文件")

        print(f"📥 [Dataset] 正在加载 {len(files)} 个 {mode} 文件...")
        
        # 2. 预加载 & 过滤
        self.global_samples = []
        dropped_count = 0
        
        for f in tqdm(files, desc="Loading PT files"):
            try:
                # 获取服务名用于查表
                # 文件名格式: service_name_train.pt
                filename = os.path.basename(f)
                service_name = filename.replace(f"_{mode}.pt", "")
                # 简化服务名匹配 (去掉 .source 等后缀，只取核心名，需与 GT 对应)
                # 你的 GT 里可能是 "cartservice"，这里可能是 "cartservice-1.source..."
                # 这是一个模糊匹配，简单起见我们尝试包含匹配
                
                content = torch.load(f, weights_only=False)
                metrics = content['metrics'] 
                logs = content['logs']       
                timestamps = content['timestamps'] # 必须有时间戳才能过滤
                
                if 'metric_mask' in content:
                    feature_mask = content['metric_mask']
                else:
                    feature_mask = torch.ones(333, dtype=torch.float)
                
                num_steps = metrics.shape[0]
                if num_steps <= metric_window: continue 
                
                for i in range(metric_window, num_steps):
                    # --- 【核心手术】 剔除故障样本 ---
                    if mode == 'train':
                        current_ts = timestamps[i]
                        is_dirty = False
                        
                        # 查找该服务是否有故障记录
                        # 简单匹配：遍历 fault_ranges 的 key，看是否包含在 filename 里
                        for fault_svc, ranges in self.fault_ranges.items():
                            if fault_svc in service_name: # 命中服务
                                for (start, end) in ranges:
                                    if start <= current_ts <= end:
                                        is_dirty = True
                                        break
                            if is_dirty: break
                        
                        if is_dirty:
                            dropped_count += 1
                            continue # 跳过这个样本，不要放入训练集！
                    # ------------------------------------

                    curr_log = logs[i]
                    if len(curr_log) == 0: continue
                    if len(curr_log) == 1 and curr_log[0] == 0: continue

                    self.global_samples.append({
                        'metric_data': metrics,      
                        'log_data': logs,            
                        'feature_mask': feature_mask,
                        'idx': i                     
                    })
                    
            except Exception as e:
                print(f"⚠️ 加载 {f} 失败: {e}")
                
        print(f"✅ {mode} 集加载完毕！")
        print(f"   -> 有效样本: {len(self.global_samples)}")
        if mode == 'train':
            print(f"   -> 🧹 成功剔除故障样本: {dropped_count} (防止模型学会'故障匹配')")

    def __len__(self):
        return len(self.global_samples)

    def __getitem__(self, index):
        # ... (保持原本的 __getitem__ 代码不变，不需要改动) ...
        # 请把之前 V7.0 dataset.py 的 __getitem__ 完整复制过来
        # 包含 log_count 计算的那部分
        sample_info = self.global_samples[index]
        curr_idx = sample_info['idx']
        
        metric_seq = sample_info['metric_data'][curr_idx - self.metric_window : curr_idx]
        feature_mask = sample_info['feature_mask']
        log_seq_raw = sample_info['log_data'][curr_idx]
        
        log_seq = torch.zeros(self.log_max_len, dtype=torch.long)
        log_mask = torch.zeros(self.log_max_len, dtype=torch.float)
        
        current_len = len(log_seq_raw)
        if current_len > 0:
            if current_len > self.log_max_len:
                current_len = self.log_max_len
                log_seq_raw = log_seq_raw[:current_len]
            if isinstance(log_seq_raw, np.ndarray):
                tensor_raw = torch.from_numpy(log_seq_raw).long()
            else:
                tensor_raw = torch.tensor(log_seq_raw).long()
            tensor_raw = tensor_raw.clamp(max=self.vocab_size - 1)
            log_seq[:current_len] = tensor_raw
            log_mask[:current_len] = 1.0 

        log_count = torch.zeros(self.vocab_size, dtype=torch.float)
        if current_len > 0:
            counts = torch.bincount(tensor_raw, minlength=self.vocab_size).float()
            log_count = counts[:self.vocab_size]
            log_count = torch.log1p(log_count) 

        if self.mode == 'train':
            neg_idx = np.random.randint(0, len(self.global_samples))
            while neg_idx == index: 
                neg_idx = np.random.randint(0, len(self.global_samples))
            neg_info = self.global_samples[neg_idx]
            neg_curr_idx = neg_info['idx']
            neg_metric_seq = neg_info['metric_data'][neg_curr_idx - self.metric_window : neg_curr_idx]
            neg_metric_mask = neg_info['feature_mask']
        else:
            neg_metric_seq = metric_seq.clone()
            neg_metric_mask = feature_mask.clone()

        return {
            'metric_seq': metric_seq,       
            'metric_mask': feature_mask,    
            'log_seq': log_seq,             
            'log_mask': log_mask,
            'log_count': log_count,
            'neg_metric_seq': neg_metric_seq, 
            'neg_metric_mask': neg_metric_mask 
        }