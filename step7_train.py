import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from step4_windowing import SMDWindowDataset
from model_v2_with_gat import MyFinalModel
import time

# === 1. 配置参数 (Hyperparameters) ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5  # 为了演示，我们先跑 5 轮。实际项目可能要跑 20-50 轮。
WINDOW_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Mac M1/M2/M3 用户的加速神器！

print(f"当前使用的计算设备: {DEVICE}")

# === 2. 准备数据 ===
print("正在加载数据...")
# 注意：我们用训练集 (train) 来训练，让模型只看正常数据
file_path = "data/ServerMachineDataset/train/machine-1-1.txt"
dataset = SMDWindowDataset(file_path, window_size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 3. 初始化模型 ===
model = MyFinalModel(num_features=36, window_size=WINDOW_SIZE).to(DEVICE)

# === 4. 定义损失函数和优化器 ===
# MSELoss: 均方误差。预测值和真实值越接近，Loss 越小。
criterion = nn.MSELoss()
# Adam: 最常用的优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === 5. 开始训练循环 (The Training Loop) ===
print("=== 开始双重任务训练 (Forecast + Reconstruction) ===")
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    start_time = time.time()  # 记录开始时间

    for i, batch in enumerate(dataloader):
        # batch: [32, 100, 36]
        x = batch.to(DEVICE)
        target_next = batch[:, -1, :].to(DEVICE)  # 预测目标：最后一个点
        target_window = x  # 重建目标：就是输入本身！

        optimizer.zero_grad()

        # --- 前向传播 (现在有3个输出了) ---
        pred_next, recon_window, _ = model(x)

        # --- 计算双重 Loss ---
        # 1. 预测 Loss
        loss_forecast = criterion(pred_next, target_next)

        # 2. 重建 Loss
        loss_recon = criterion(recon_window, target_window)

        # 3. 总 Loss (简单的 1:1 相加，或者加权)
        loss = loss_forecast + loss_recon

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (i + 1) % 100 == 0:
            # 打印详细一点，看看两个任务分别是多少
            print(
                f"Epoch [{epoch + 1}], Step [{i + 1}] | Total: {loss.item():.4f} (For: {loss_forecast.item():.4f}, Rec: {loss_recon.item():.4f})")

    avg_loss = epoch_loss / len(dataloader)
    #print(f"Epoch [{epoch + 1}] Done. Avg Loss: {avg_loss:.6f}")
    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time

    print(f"Epoch [{epoch + 1}] 完成 | 耗时: {duration:.2f} 秒 | Avg Loss: {avg_loss:.6f}")

# === 6. 保存模型 (Save the Baby) ===
# 这一步很重要，保存下来，下次就不用重新练了
torch.save(model.state_dict(), "my_trained_model.pth")
print("模型参数已保存至 my_trained_model.pth")