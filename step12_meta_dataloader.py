import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MetaTaskSampler:
    """
    元学习专用任务采样器 (Task Generator)
    这里的 'Task' 定义为：针对某一台特定机器 (Machine ID) 的异常检测/重建任务。
    """
    def __init__(self, data_dir, window_size=100, k_shot=10, q_query=10):
        """
        :param data_dir: 训练数据目录 (例如 data/ServerMachineDataset/train)
        :param k_shot: Support Set 的大小 (也就是模型能看到多少条数据来学习新机器)
        :param q_query: Query Set 的大小 (用于评估学习效果)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.k_shot = k_shot
        self.q_query = q_query
        
        # 获取所有机器的文件名列表
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        self.file_list.sort()
        print(f"🧠 Meta-Loader 初始化: 发现 {len(self.file_list)} 个机器任务。")
        
        # 预加载所有数据到内存 (SMD数据不大，为了速度可以全读进去)
        # 格式: self.data_cache['machine-1-1.txt'] = numpy array [Time, 38]
        self.data_cache = {}
        self._preload_data()

    def _preload_data(self):
        print("⏳ 正在预加载数据到内存以加速采样...")
        for f in self.file_list:
            path = os.path.join(self.data_dir, f)
            try:
                df = pd.read_csv(path, header=None)
                data = df.values.astype(np.float32)
                
                # === 🛠️ 优化 1: 维度对齐 ===
                # 如果数据多于 36 列，我们强制截取前 36 列 (SMD 常用做法)
                # 必须与你 Step 9 模型的 input_dim 一致
                if data.shape[1] > 36:
                    data = data[:, 0:36] 
                
                # === 🛠️ 优化 2: 实例归一化 (Instance Normalization) ===
                # 元学习中，每台机器的数据分布可能不同
                # 我们对每台机器单独做 MinMax，这能极大帮助模型快速适应
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                # 防止除以 0
                data = (data - min_val) / (max_val - min_val + 1e-8)
                
                self.data_cache[f] = data
            except Exception as e:
                print(f"⚠️ 无法读取 {f}: {e}")
        print("✅ 预加载完成 (已应用: 维度截断36 + MinMax归一化)！")

    def get_batch(self, meta_batch_size=4):
        """
        生成一个 Meta-Batch。
        :param meta_batch_size: 一次采样多少个任务 (多少台机器)
        :return: supports, queries
            - supports: [MetaBatch, K_Shot, Window, Feat]
            - queries:  [MetaBatch, Q_Query, Window, Feat]
        """
        # 1. 随机抽 meta_batch_size 台机器
        selected_files = np.random.choice(self.file_list, meta_batch_size, replace=False)
        
        support_x_batch = []
        query_x_batch = []
        
        for filename in selected_files:
            data = self.data_cache[filename] # [Time, Feat]
            total_len = len(data)
            
            # 我们需要切出 K + Q 个样本，每个样本长度为 window_size
            # 为了保证不越界，最大起始点是:
            max_start_idx = total_len - self.window_size
            
            if max_start_idx <= 0:
                continue # 数据太短，跳过
            
            # 2. 在这台机器上随机切片
            # 我们需要 k_shot + q_query 个不重复的起始点
            # 也可以简单点：随机选一段连续的时间，前 K 给 Support，后 Q 给 Query (更符合实际场景)
            
            # 方案 A: 随机离散采样 (更难，泛化性更强)
            # start_indices = np.random.choice(max_start_idx, self.k_shot + self.q_query, replace=False)
            
            # 方案 B: 连续采样 (模拟：拿前 K 分钟学习，预测后 Q 分钟) -> 我们选这个，更符合运维逻辑
            # 随机选一个起点
            segment_len = self.k_shot + self.q_query
            # 确保连续取样不越界，且为了简单，我们让样本之间不重叠(stride=window)或者步长为1
            # 这里为了数据多样性，我们在整条时间轴上随机采 K+Q 个点
            start_indices = np.random.choice(max_start_idx, self.k_shot + self.q_query, replace=False)
            
            machine_samples = []
            for start in start_indices:
                window = data[start : start + self.window_size, :]
                machine_samples.append(window)
            
            machine_samples = np.array(machine_samples) # [K+Q, Window, Feat]
            
            # 切分 Support 和 Query
            support_x = machine_samples[:self.k_shot]
            query_x = machine_samples[self.k_shot:]
            
            support_x_batch.append(support_x)
            query_x_batch.append(query_x)
            
        # 转 tensor
        # Shape: [MetaBatch, K, Window, Feat]
        support_x_batch = torch.from_numpy(np.array(support_x_batch))
        query_x_batch = torch.from_numpy(np.array(query_x_batch))
        
        return support_x_batch, query_x_batch

# === 测试代码 (Test Block) ===
if __name__ == "__main__":
    # 配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_dir, 'data', 'ServerMachineDataset', 'train')
    
    # 初始化采样器
    # 假设我们要模拟：看 5 个样本 (5-Shot)，预测 5 个样本
    sampler = MetaTaskSampler(train_dir, window_size=100, k_shot=5, q_query=5)
    
    # 获取一个 Batch
    supports, queries = sampler.get_batch(meta_batch_size=2)
    
    print("\n✅ 采样成功！")
    print(f"Supports Shape: {supports.shape} -> [MetaBatch, K_Shot, Window, Features]")
    print(f"Queries  Shape: {queries.shape} -> [MetaBatch, Q_Query, Window, Features]")
    
    print("\n含义解释:")
    print("Supports: 模型用来'突击复习'的课本。")
    print("Queries : 模型复习完后用来'考试'的真题。")