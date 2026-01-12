import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os  # <--- 必不可少的模块

# 定义一个类，继承自 PyTorch 的 Dataset
class SMDDataset(Dataset):
    def __init__(self, file_path):
        """
        初始化函数
        :param file_path: 数据文件的绝对路径
        """
        # === 增加健壮性检查 ===
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ [Error] 找不到文件: {file_path}")
            
        # 初始化：读取数据，转成 float32 (神经网络喜欢 float32)
        # 提示：values 会把 pandas DataFrame 转成 numpy array
        self.data = pd.read_csv(file_path, header=None).values.astype(np.float32)

    def __len__(self):
        # 告诉 PyTorch 数据有多少条
        return len(self.data)

    def __getitem__(self, idx):
        # 告诉 PyTorch 怎么取第 idx 条数据
        return self.data[idx]


# --- 测试一下我们的类 ---
if __name__ == "__main__":
    # ==========================================
    # 核心修改：自动构建绝对路径
    # ==========================================
    # 1. 获取当前脚本所在文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 拼接测试文件的路径
    # 目标: data/ServerMachineDataset/test/machine-1-1.txt
    file_path = os.path.join(current_dir, 'data', 'ServerMachineDataset', 'test', 'machine-1-1.txt')
    
    print(f"🔍 [Debug] 正在尝试读取: {file_path}")

    try:
        # 1. 实例化数据集
        dataset = SMDDataset(file_path)
        print(f"✅ 数据集实例化成功！共 {len(dataset)} 条数据。")

        # 2. 创建 DataLoader (加载器)
        # batch_size=32 意味着一次喂给模型 32 行数据
        # shuffle=True 意味着打乱顺序训练 (训练集通常要打乱)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 3. 模拟一次训练循环
        print("=== 开始模拟 PyTorch 数据加载 ===")
        for i, batch in enumerate(dataloader):
            print(f"📦 第 {i+1} 批数据 (Batch)")
            print(f"   形状: {batch.shape}")  # 应该是 [32, 38]
            print(f"   类型: {batch.dtype}")

            # 我们只打印第一批就停，证明跑通了即可
            break

        print("\n=== 🎉 成功！你的数据已经可以喂给神经网络了 ===")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")