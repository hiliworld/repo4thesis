import torch
from step4_windowing import SMDWindowDataset
from torch.utils.data import DataLoader
from lnt_model import LNT_Encoder  # 导入刚才写的模型

# 1. 准备数据
file_path = "/Users/chariesliu/Desktop/MyThesis/ServerMachineDataset/test/machine-1-1.txt"
dataset = SMDWindowDataset(file_path, window_size=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 准备模型
# input_dim=36 (因为你只有36列数据)
model = LNT_Encoder(input_dim=36, hidden_dim=64, z_dim=16)

# 3. 试运行
print("=== 开始模型联调测试 ===")
for batch in dataloader:
    print(f"输入数据形状: {batch.shape}")  # [32, 100, 36]

    # 前向传播 (Forward Pass)
    z_vector = model(batch)

    print(f"模型输出(Z向量)形状: {z_vector.shape}")  # 应该是 [32, 16]

    if z_vector.shape == (32, 16):
        print("✅ 完美！模型成功吃进了数据，并吐出了特征向量！")
        print("接下来就可以把这个 z_vector 喂给 GAT 了！")
    else:
        print("❌ 模型输出尺寸不对")

    break