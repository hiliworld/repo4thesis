import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

# === 导入自定义模块 ===
try:
    from step4_windowing import SMDWindowDataset
    from model_v2_with_gat import MyFinalModel
    from contrastive import ContrastiveLoss 
    from lnt_encoder import LNT_Conv_Encoder
except ImportError:
    print("❌ 错误：找不到模块，请检查目录下是否有 step4_windowing.py, model_v2_with_gat.py, contrastive.py, lnt_encoder.py")
    exit()

# === 1. 配置参数 (Hyperparameters) ===
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100       # ✅ 修改：增加轮数，给自适应图学习更多时间
WINDOW_SIZE = 100
FEATURE_DIM = 36
LAMBDA_CL = 0.1
PATIENCE = 10      # ✅ 新增：早停容忍度 (10轮不降就停)
MIN_DELTA = 0.001  # ✅ 新增：哪怕你降了，但没降够 0.001，我也不认！

# 自动选择设备
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🔥 当前使用的计算设备: {DEVICE}")

# === 2. 准备数据 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data_dir = os.path.join(current_dir, 'data', 'ServerMachineDataset', 'train')
# ✅ 修改：定义模型保存路径 (为了不覆盖旧模型，我们改个名)
save_path = os.path.join(current_dir, "my_trained_model_adaptive.pth")

print(f"📂 正在加载训练数据: {train_data_dir}")

def data_augmentation(x):
    """
    简单的数据增强：随机抖动 (Jittering) 和 缩放 (Scaling)
    x: [Batch, Window, Features]
    """
    noise = torch.normal(0, 0.01, size=x.shape).to(x.device)
    scale = torch.normal(1, 0.1, size=x.shape).to(x.device)
    return x * scale + noise

try:
    train_dataset = SMDWindowDataset(train_data_dir, window_size=WINDOW_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"✅ 数据加载成功！共 {len(train_dataset)} 个样本。")
    print(f"   每个 Epoch 将迭代 {len(train_loader)} Steps。")

except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit()

# === 3. 初始化模型 ===
model = MyFinalModel(
    num_features=FEATURE_DIM, 
    window_size=WINDOW_SIZE,
    hidden_dim=64, 
    z_dim=16
).to(DEVICE)

# === 4. 定义损失函数和优化器 ===
criterion_mse = nn.MSELoss()
criterion_cl = ContrastiveLoss(BATCH_SIZE, device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === 5. 开始训练循环 ===
print("\n=== 🚀 开始 Step 9 完全体训练 (Adaptive GAT + Contrastive) ===")
print(f"   (CL权重: {LAMBDA_CL}, Max Epochs: {EPOCHS}, Early Stopping: {PATIENCE})")

model.train()
total_start_time = time.time()

# 早停相关变量
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_cl_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        x = batch.to(DEVICE)
        
        # 1. 准备 Target
        target_next = x[:, -1, :]
        target_window = x

        optimizer.zero_grad()

        # ==========================
        # 🛣️ 路径 A: 主任务 (MSE)
        # ==========================
        # 注意：这里 _ 接住的是 attention_weights，训练时不需要用到
        pred_next, recon_window, _ = model(x)
        
        loss_forecast = criterion_mse(pred_next, target_next)
        loss_recon = criterion_mse(recon_window, target_window)
        loss_main = loss_forecast + loss_recon

        # ==========================
        # 🛣️ 路径 B: 对比学习 (CL)
        # ==========================
        x_i = data_augmentation(x)
        x_j = data_augmentation(x)
        
        # 只用 Encoder 提取特征 (高效)
        z_i = model.lnt_encoder(x_i)
        z_j = model.lnt_encoder(x_j)
        
        z_i_flat = z_i.view(BATCH_SIZE, -1)
        z_j_flat = z_j.view(BATCH_SIZE, -1)
        
        loss_cl = criterion_cl(z_i_flat, z_j_flat)

        # ==========================
        # ⚖️ 总损失融合
        # ==========================
        loss_total = loss_main + (LAMBDA_CL * loss_cl)

        loss_total.backward()
        optimizer.step()

        epoch_loss += loss_total.item()
        epoch_cl_loss += loss_cl.item()

        # 打印日志 (每 10% 进度)
        log_interval = max(10, len(train_loader) // 10)
        if (i + 1) % log_interval == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] "
                  f"| Total: {loss_total.item():.4f} "
                  f"(MSE: {loss_main.item():.4f}, CL: {loss_cl.item():.4f})")

    # Epoch 结束统计
    avg_loss = epoch_loss / len(train_loader)
    avg_cl = epoch_cl_loss / len(train_loader)
    duration = time.time() - start_time
    
    print(f"✨ Epoch [{epoch + 1}] Done | Time: {duration:.1f}s | Avg Loss: {avg_loss:.4f} (Avg CL: {avg_cl:.4f})")

    # === ✅ 早停逻辑 (Early Stopping) ===
    # === ✅ 升级版早停逻辑 ===
    # 只有当 (当前Loss < 历史最佳 - 阈值) 时，才算有效进步
    if avg_loss < (best_loss - MIN_DELTA):
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"   💾 Loss 有显著下降 (超过 {MIN_DELTA})! 模型已保存。")
        
    else:
        # 即使 avg_loss 比 best_loss 小一点点 (比如 0.0001)，只要没超过 MIN_DELTA，也算没进步
        patience_counter += 1
        print(f"   ⚠️ Loss 进入平台期 ({patience_counter}/{PATIENCE}) - 当前: {avg_loss:.4f}, 最佳: {best_loss:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"🛑 触发早停！模型在 {PATIENCE} 轮内没有显著提升 (>{MIN_DELTA})。")
            break

print("="*50)
print(f"✅ 训练结束！总耗时: {(time.time() - total_start_time)/60:.1f} 分钟")
print(f"   最终最佳 Loss: {best_loss:.4f}")
print(f"   模型已保存至: {save_path}")