import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class HDFSDataset(Dataset):
    def __init__(self, structured_file, label_file, mode='train', max_len=50, vocab_map=None):
        """
        :param structured_file: 解析后的日志序列 (BlockId, EventSequence)
        :param label_file: 标签文件 (BlockId, Label)
        :param mode: 'train' 或 'test'
        :param max_len: 序列最大长度 (Padding/Truncation)
        """
        self.max_len = max_len
        self.mode = mode
        
        print(f"🔄 [{mode.upper()}] 正在加载 HDFS 数据...")
        
        # 1. 读取数据
        # 这里的 engine='python' 是为了防止某些特殊字符报错
        df_logs = pd.read_csv(structured_file, engine='python')
        df_labels = pd.read_csv(label_file, engine='python')
        
        # 2. 合并数据 (Merge)
        # 只有既有日志又有标签的 Block 才能用于训练/测试
        self.data_df = pd.merge(df_logs, df_labels, on='BlockId', how='inner')
        
        # 处理标签: Normal -> 0, Anomaly -> 1
        self.data_df['Label'] = self.data_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
        
        print(f"   原始样本数: {len(self.data_df)} (异常率: {self.data_df['Label'].mean():.2%})")

        # 3. 构建词汇表 (Vocabulary)
        # 我们需要知道一共有多少种 Event ID
        all_events = set()
        for seq in self.data_df['EventSequence']:
            # seq 是字符串 "1 5 2"，需要 split
            if pd.isna(seq): continue
            events = seq.split()
            all_events.update(events)
            
        # 0 号留给 Padding，所以从 1 开始映射
        # 如果是训练集，我们需要建立映射表；如果是测试集，必须复用训练集的映射表
        if vocab_map is None:
            sorted_events = sorted(list(all_events), key=lambda x: int(x))
            self.vocab_map = {event: i+1 for i, event in enumerate(sorted_events)}
            print(f"   🆕 构建新词汇表: {len(self.vocab_map)} 个唯一事件 (ID 1-{len(self.vocab_map)})")
        else:
            self.vocab_map = vocab_map
            print(f"   ♻️ 复用词汇表: {len(self.vocab_map)} 个事件")

        self.vocab_size = len(self.vocab_map) + 1 # +1 是因为有 0 (Padding)

        # 4. 序列数字化与定长化
        self.sequences = []
        self.labels = self.data_df['Label'].values
        
        for seq in self.data_df['EventSequence']:
            if pd.isna(seq):
                idxs = []
            else:
                # 将字符串 "1 5" 转为数字 [1, 5]，未知的转为 0 (UNK)
                # get(e, 0) 这里 0 其实不太好，通常我们用单独的 UNK token，但在日志里很少有未知
                idxs = [self.vocab_map.get(e, 0) for e in seq.split()]
            
            # === Padding / Truncation ===
            if len(idxs) < self.max_len:
                # 太短：前面补 0 (Pre-padding) 或者后面补 0 (Post-padding)
                # 对于 LSTM/Transformer，通常建议补 0
                padded = [0] * (self.max_len - len(idxs)) + idxs
            else:
                # 太长：截取最后 max_len 个 (因为异常通常在最后)
                padded = idxs[-self.max_len:]
            
            self.sequences.append(padded)
            
        self.sequences = np.array(self.sequences)
        print(f"✅ 数据准备完毕. Shape: {self.sequences.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 返回: (序列输入, 标签)
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# === 工厂函数 ===
def get_hdfs_loaders(data_dir='data/HDFS/HDFS_v1', batch_size=64, test_ratio=0.2):
    """
    自动划分训练集和测试集
    """
    import os
    structured_path = os.path.join(data_dir, 'hdfs_structured.csv')
    label_path = os.path.join(data_dir, 'anomaly_label.csv') # 也就是 ground truth
    
    # 1. 先实例化一个全量数据集 (为了拿 vocab_map)
    # 这里我们偷个懒，先读一遍全量，然后再 split
    # 实际生产中应该只读 train 部分建立 vocab
    full_dataset = HDFSDataset(structured_path, label_path, mode='full', max_len=50)
    
    # 2. 划分索引
    # HDFS 通常是按时间顺序，但这里为了简单先随机划分
    # stratify=labels 保证训练集和测试集的异常比例一致
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=test_ratio, 
        random_state=42, 
        stratify=full_dataset.labels
    )
    
    # 3. 创建子集 (Subset)
    # PyTorch 的 Subset 非常好用，不需要重新加载数据
    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    test_set = torch.utils.data.Subset(full_dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"✂️ 数据集划分: 训练集 {len(train_set)}, 测试集 {len(test_set)}")
    
    return train_loader, test_loader, full_dataset.vocab_size

# === 测试代码 ===
if __name__ == "__main__":
    # 简单的测试运行
    train_loader, test_loader, vocab_size = get_hdfs_loaders()
    
    # 拿一个 batch 看看长什么样
    for x, y in train_loader:
        print("\n🔎 Batch Preview:")
        print(f"Input Shape: {x.shape} (Batch, MaxLen)")
        print(f"Label Shape: {y.shape}")
        print(f"Sample Sequence: {x[0]}") # 看看是不是有 0 (Padding)
        print(f"Vocab Size: {vocab_size}")
        break