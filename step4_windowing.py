import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# === 这是一个工业级的时序数据加载类 ===
class SMDWindowDataset(Dataset):
    def __init__(self, file_path, window_size=100):
        """
        初始化函数
        :param file_path: 数据文件路径
        :param window_size: 窗口大小 (比如过去100个时间点)
        """
        # 1. 读取数据
        raw_data = pd.read_csv(file_path, header=None).values.astype(np.float32)

        # 2. 【关键优化】去除全是0的列 (根据你的运行结果，最后两列是坏的)
        # 我们只保留前36列 (索引 0 到 35)
        self.data = raw_data[:, :-2]
        print(f"原始形状: {raw_data.shape} -> 优化后形状: {self.data.shape}")

        self.window_size = window_size

    def __len__(self):
        # 能够切出多少个窗口？
        # 举例：如果你有 105 分钟数据，窗口是 100，那你只能切出 (105-100) = 5 个窗口
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        """
        核心逻辑：怎么拿第 idx 个窗口？
        如果是 idx=0，拿 0~100 行
        如果是 idx=1，拿 1~101 行
        """
        # 截取从 idx 开始，长度为 window_size 的一段数据
        window_data = self.data[idx: idx + self.window_size]

        # 返回这一段数据 (PyTorch 会自动把它转成 Tensor)
        return window_data


# === 测试代码 ===
if __name__ == "__main__":
    # 使用相对路径
    file_path = "/Users/chariesliu/Desktop/MyThesis/ServerMachineDataset/test/machine-1-1.txt"

    # 1. 实例化：设定窗口大小为 100 (这是LNT论文常用的设置)
    dataset = SMDWindowDataset(file_path, window_size=100)

    # 2. 创建 DataLoader
    # batch_size=32 意味着一次喂给模型 32 个“窗口”
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. 看看吐出来的数据长什么样
    print("\n=== 开始检查数据格式 ===")
    for batch in dataloader:
        # batch 的形状应该是 [32, 100, 36]
        # 32: 一批有多少个样本
        # 100: 每个样本的时间长度 (Seq_Len)
        # 36: 特征数量 (Features) - 注意不是38了，因为去掉了两列
        print(f"输入模型的 Tensor 形状: {batch.shape}")

        if batch.shape == (32, 100, 36):
            print("✅ 成功！这个形状可以直接喂给 LNT 和 GAT 模型了！")
        else:
            print("❌ 形状不对，请检查代码。")

        break  # 只看第一批