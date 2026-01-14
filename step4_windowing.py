import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob

# === 工业级：支持目录读取与单文件读取 ===
class SMDWindowDataset(Dataset):
    def __init__(self, path, window_size=100):
        """
        初始化函数
        :param path: 可以是【文件夹路径】(训练时用)，也可以是【单个文件路径】(测试时用)
        :param window_size: 时间窗口大小
        """
        self.window_size = window_size
        self.windows_list = [] # 用于存储切好的窗口
        
        # 1. 智能判断输入是文件夹还是文件
        if os.path.isdir(path):
            # 如果是文件夹，读取下面所有 txt
            print(f"📂 [Dataset] 检测到目录，正在扫描: {path}")
            file_pattern = os.path.join(path, "*.txt")
            file_paths = sorted(glob.glob(file_pattern))
        elif os.path.isfile(path):
            # 如果是文件，只读取这一个
            print(f"📄 [Dataset] 检测到单文件，正在加载: {path}")
            file_paths = [path]
        else:
            raise FileNotFoundError(f"❌ 路径不存在: {path}")

        if len(file_paths) == 0:
            raise ValueError(f"❌ 在路径下没找到任何 .txt 文件！")

        # 2. 循环处理每个文件 (关键：避免跨文件污染)
        total_samples = 0
        
        for p in file_paths:
            try:
                # A. 读取单文件
                df = pd.read_csv(p, header=None)
                raw_data = df.values.astype(np.float32)
                
                # B. 特征裁剪 (只取前36列)
                # 确保所有机器都做同样的处理
                data = raw_data[:, :-2] 
                
                # C. 单独切窗口
                # 只有当数据长度大于窗口大小时才能切
                num_samples = len(data) - window_size
                if num_samples > 0:
                    # 使用列表推导式快速切分
                    # 这里的逻辑是：对于当前这个文件，切出所有可能的窗口
                    current_file_windows = [data[i : i+window_size] for i in range(num_samples)]
                    
                    # 将这些窗口加入总列表
                    self.windows_list.extend(current_file_windows)
                    total_samples += num_samples
                    
            except Exception as e:
                print(f"⚠️ 读取文件出错 {p}: {e}")

        # 3. 转换为 Numpy 数组 (方便 PyTorch 处理)
        if len(self.windows_list) == 0:
            raise ValueError("❌ 数据集为空，未能生成任何窗口！可能文件太小或格式不对。")
            
        self.data = np.array(self.windows_list)
        print(f"✅ 数据加载完成！最终形状: {self.data.shape} (占用内存约 {self.data.nbytes / 1024 / 1024:.2f} MB)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 把 numpy 转成 tensor
        return torch.from_numpy(self.data[idx])


# === 测试代码 ===
if __name__ == "__main__":
    # 简单测试一下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 找个文件测一下
    test_file = os.path.join(current_dir, 'data', 'ServerMachineDataset', 'test', 'machine-1-1.txt')
    
    if os.path.exists(test_file):
        try:
            ds = SMDWindowDataset(test_file, window_size=100)
            print("单文件测试通过！")
        except Exception as e:
            print(f"测试失败: {e}")
    else:
        print("没找到测试文件，跳过测试。")