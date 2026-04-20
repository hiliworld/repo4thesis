import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

class SmartTimeSeriesDataset(Dataset):
    def __init__(self, data_path, config, mode='train', scaler_stats=None, clean_stats=None):
        """
        :param scaler_stats: 全局归一化参数 (min, max)
        :param clean_stats: 全局清洗参数 (保留哪些列的 index)
        """
        self.config = config
        self.mode = mode
        
        # === 1. 扫描文件 (File Scanning) ===
        self.file_paths = self._scan_files(data_path)
        print(f"[{mode.upper()}] 扫描到 {len(self.file_paths)} 个数据文件: {data_path}")

        # === 2. 加载所有数据 (Load All) ===
        # 为了保证清洗和归一化的一致性，我们需要先把所有数据加载进来（内存允许的情况下）
        # 对于超大数据集，这里需要改为流式处理，但 SMD/SWaT 都可以放入内存
        self.df_list = []
        for fp in self.file_paths:
            df = self._load_single_file(fp)
            self.df_list.append(df)
            
        # 拼接成一个巨大的临时表来计算统计量 (只用于计算，不用于切窗)
        full_df = pd.concat(self.df_list, axis=0, ignore_index=True)
        print(f"   原始数据总量: {full_df.shape}")

        # === 3. 全局自动清洗 (Global Auto-Cleaning) ===
        # 核心逻辑：训练集决定哪些列要留，测试集必须遵守！
        if mode == 'train':
            if config['dataset']['cleaning']['auto_clean']:
                # 计算保留列的索引
                self.keep_cols = self._get_clean_columns(full_df)
            else:
                self.keep_cols = list(range(full_df.shape[1]))
        else:
            # 测试集：直接复用训练集的列策略
            self.keep_cols = clean_stats['keep_cols']

        # 应用列筛选
        # 注意：这里我们是对 df_list 里的每个 df 单独操作，而不是对 full_df 操作
        # 这样保留了文件边界
        cleaned_list = [df.iloc[:, self.keep_cols] for df in self.df_list]
        
        # 重新生成 full_df 用于计算归一化参数
        full_cleaned = pd.concat(cleaned_list, axis=0, ignore_index=True)
        self.feature_dim = full_cleaned.shape[1]
        
        print(f"   清洗后特征维度: {self.feature_dim} (丢弃了 {full_df.shape[1] - self.feature_dim} 列)")

        # === 4. 全局归一化 (Global Normalization) ===
        if config['dataset']['normalization'] == 'minmax':
            if mode == 'train':
                self.min_val = np.nanmin(full_cleaned.values, axis=0)
                self.max_val = np.nanmax(full_cleaned.values, axis=0)
                self.scale_denom = self.max_val - self.min_val + 1e-8
            else:
                self.min_val = scaler_stats['min']
                self.max_val = scaler_stats['max']
                self.scale_denom = scaler_stats['denom']

        # === 5. 独立切窗 (Safe Windowing) ===
        self.window_size = config['dataset']['window_size']
        self.windows = []
        self.full_sequences = []
        
        # 逐个文件处理，绝不跨文件切窗！
        for df in cleaned_list:
            # 1. 填补 NaN (线性插值)
            df = df.interpolate(method='linear', limit_direction='both').fillna(0)
            data = df.values.astype(np.float32)
            
            # 2. 归一化 (关键：运算后必须强转回 float32，否则会变成 float64)
            data = ((data - self.min_val) / self.scale_denom).astype(np.float32)
            self.full_sequences.append(data)
            
            # 3. 切片
            if len(data) >= self.window_size:
                for i in range(len(data) - self.window_size):
                    self.windows.append(data[i : i + self.window_size])
        
        # 转 numpy (加速 DataLoader)
        self.windows = np.array(self.windows)
        print(f"[{mode.upper()}] 准备就绪. 生成窗口数: {len(self.windows)}")

    def _scan_files(self, path):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            pattern = self.config['dataset']['format'].get('pattern', '*.txt')
            # 递归查找或只查找当前层
            search_path = os.path.join(path, pattern)
            files = sorted(glob.glob(search_path))
            if not files:
                raise FileNotFoundError(f"在 {path} 下没找到 {pattern} 文件")
            return files
        else:
            raise FileNotFoundError(f"路径不存在: {path}")

    def _load_single_file(self, path):
        fmt = self.config['dataset']['format']
        sep = fmt.get('sep', ',')
        header = fmt.get('header', None)
        
        try:
            df = pd.read_csv(path, sep=sep, header=header)
        except:
            df = pd.read_csv(path, sep=r'\s+', header=header)
            
        # 分离标签
        label_col = fmt.get('label_col', None)
        if label_col is not None:
            if label_col < 0: label_col = df.columns[label_col]
            df = df.drop(columns=[label_col])
            
        # 强制转数值
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    def _get_clean_columns(self, full_df):
        """
        基于全部数据的统计特征，决定保留哪些列的索引
        """
        clean_conf = self.config['dataset']['cleaning']
        
        # 1. 统计缺失率
        threshold = clean_conf.get('drop_high_nan', 0.5)
        nan_ratio = full_df.isna().mean()
        keep_mask = nan_ratio < threshold
        
        # 2. 统计方差 (排除常数列)
        if clean_conf.get('drop_constant', True):
            std = full_df.std()
            # 只有当标准差 > 0 (或者 NaN) 时才保留
            # 注意：全NaN的列 std 是 NaN，前面已经处理过高缺失率了，这里主要看 std==0
            keep_mask = keep_mask & (std > 1e-6)
            
        # 返回 True 的列索引
        return full_df.columns[keep_mask].tolist()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])

    def get_full_sequences(self):
        """返回归一化后的完整序列列表，供在线滚动推理使用。"""
        return self.full_sequences

    def normalize_external_sequence(self, seq):
        """对外部序列复用训练归一化统计量。"""
        seq = np.asarray(seq, dtype=np.float32)
        return ((seq - self.min_val) / self.scale_denom).astype(np.float32)

    def denormalize_sequence(self, seq):
        seq = np.asarray(seq, dtype=np.float32)
        return (seq * self.scale_denom + self.min_val).astype(np.float32)

def get_dataloaders(config_path='config.yaml', return_datasets=False):
    # 统一使用 UTF-8（兼容 BOM），避免在 Windows 默认编码下读取失败
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    
    # 1. 加载训练集 (计算全局统计量)
    train_dataset = SmartTimeSeriesDataset(
        config['dataset']['train_file'], 
        config, 
        mode='train'
    )
    
    # 打包统计量传给测试集
    scaler_stats = {
        'min': train_dataset.min_val,
        'max': train_dataset.max_val,
        'denom': train_dataset.scale_denom
    }
    clean_stats = {
        'keep_cols': train_dataset.keep_cols
    }
    
    # 2. 加载测试集 (复用统计量)
    test_dataset = SmartTimeSeriesDataset(
        config['dataset']['test_file'], 
        config, 
        mode='test',
        scaler_stats=scaler_stats,
        clean_stats=clean_stats
    )
    
    #train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    # drop_last=True 是为了防止最后一个 batch 大小不一致导致 shape 错误
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        drop_last=True  # <--- 关键修改
    )
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)
    
    if return_datasets:
        return train_loader, test_loader, train_dataset.feature_dim, train_dataset, test_dataset
    return train_loader, test_loader, train_dataset.feature_dim
