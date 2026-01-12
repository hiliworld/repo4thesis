import torch
from step4_windowing import SMDWindowDataset
from torch.utils.data import DataLoader
from model_v2_with_gat import MyFinalModel

# 1. 准备数据
file_path = "/Users/chariesliu/Desktop/MyThesis/ServerMachineDataset/test/machine-1-1.txt"
dataset = SMDWindowDataset(file_path, window_size=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 实例化模型
model = MyFinalModel(num_features=36)

print("=== 开始 GAT 架构测试 ===")
for batch in dataloader:
    print(f"输入形状: {batch.shape}")

    # 前向传播
    score, attention_map = model(batch)

    print(f"最终评分形状: {score.shape}")  # 应该是 [32, 1]
    print(f"注意力图形状: {attention_map.shape}")  # 应该是 [32, 36, 36]

    if attention_map.shape == (32, 36, 36):
        print("✅ 成功！你已经成功构建了图神经网络！")
        print("注意看：这个 36x36 的矩阵，就是指标之间的关系图。")
        print("比如 attention_map[0, 0, 1] 就是 'CPU' 对 '内存' 的关注度。")

    break