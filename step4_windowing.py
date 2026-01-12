import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob

# === 工业级：支持目录读取与自动路径 ===
class SMDWindowDataset(Dataset):
    def __init__(self, data_dir, window_size=100):
        """
        初始化函数
        :param data_dir: 数据文件夹路径 (例如 .../train)
        :param window_size: 时间窗口大小
        """
        self.window_size = window_size
        self.windows_list = [] # 用于存储切好的窗口
        
        # 1. 检查路径
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ 目录不存在: {data_dir}")
            
        # 2. 获取所有 .txt 文件
        # os.path.join(data_dir, "*.txt") 会拼出如 ".../train/*.txt"
        file_pattern = os.path.join(data_dir, "*.txt")
        file_paths = sorted(glob.glob(file_pattern))
        
        if len(file_paths) == 0:
            raise ValueError(f"❌ 在 {data_dir} 下没找到 .txt 文件！")
            
        print(f"📂 正在扫描 {data_dir}...")
        print(f"   发现 {len(file_paths)} 个文件，开始逐个处理...")

        # 3. 循环处理每个文件 (关键：避免跨文件污染)
        total_samples = 0
        
        for path in file_paths:
            try:
                # A. 读取单文件
                df = pd.read_csv(path, header=None)
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
                print(f"⚠️ 读取文件出错 {path}: {e}")

        # 4. 转换为 Numpy 数组 (方便 PyTorch 处理)
        # 此时 self.data 的形状应该是 [总样本数, 100, 36]
        if len(self.windows_list) == 0:
            raise ValueError("❌ 数据集为空，未能生成任何窗口！")
            
        self.data = np.array(self.windows_list)
        print(f"✅ 数据加载完成！最终形状: {self.data.shape} (占用内存约 {self.data.nbytes / 1024 / 1024:.2f} MB)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 把 numpy 转成 tensor
        # 注意：这里不需要再切片了，因为我们在 __init__ 里已经切好了
        return torch.from_numpy(self.data[idx])


# === 测试代码 ===
if __name__ == "__main__":
    # 1. 自动获取绝对路径 (服务器适配)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 指向测试数据文件夹
    # 注意：为了快速测试，我们还是指向 test 文件夹，或者你可以指向 train
    test_data_dir = os.path.join(current_dir, 'data', 'ServerMachineDataset', 'test')
    
    print(f"🔍 [Debug] 目标文件夹: {test_data_dir}")

    try:
        # 3. 实例化 (此时会读取文件夹下所有的 txt)
        # 注意：如果 test 文件夹里文件很多，这一步可能会花几秒钟
        dataset = SMDWindowDataset(test_data_dir, window_size=100)

        # 4. 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 5. 检查 Batch
        print("\n=== 🚀 开始检查 Batch 格式 ===")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} 形状: {batch.shape}")
            
            # 校验形状 [Batch, Window, Feature]
            if batch.shape[1:] == (100, 36):
                print("✅ 校验通过：(100, 36)")
            else:
                print(f"❌ 校验失败：期望 (..., 100, 36)，实际 {batch.shape}")
            
            # 只看前两个 batch，避免刷屏
            if i >= 1:
                break
                
        print("\n🎉 Step 4 升级完成！现在支持多文件读取了！")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")