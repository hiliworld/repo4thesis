import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import yaml  

# === 导入自定义模块 ===
try:
    from data_factory import get_dataloaders
    from model_v2_with_gat import MyFinalModel
    from contrastive import ContrastiveLoss 
except ImportError:
    print("❌ 错误：找不到模块，请检查目录下是否有 data_factory.py, model_v2_with_gat.py 等文件")
    exit()

# === 1. 加载配置文件 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')

if not os.path.exists(config_path):
    print(f"❌ 错误：找不到配置文件 {config_path}")
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# === 提取参数 ===
BATCH_SIZE = config['train']['batch_size']
LEARNING_RATE = float(config['train']['lr'])
EPOCHS = config['train']['epochs']
PATIENCE = config['train']['patience']
WINDOW_SIZE = config['dataset']['window_size']
LAMBDA_CL = 0.1
MIN_DELTA = 0.001

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的计算设备: {DEVICE}")

# === 2. 准备数据 ===
save_path = os.path.join(current_dir, "my_trained_model_adaptive.pth")
print("📂 正在通过工厂加载数据...")

try:
    train_loader, val_loader, input_dim = get_dataloaders(config_path)
    print(f"✅ 数据加载成功！特征维度: {input_dim}")
except Exception as e:
    print(f"❌ 数据工厂加载失败: {e}")
    exit()

# === 数据增强 ===
def data_augmentation(x):
    noise = torch.normal(0, 0.01, size=x.shape).to(x.device)
    scale = torch.normal(1, 0.1, size=x.shape).to(x.device)
    return x * scale + noise

# === 3. 初始化模型 ===
# 更新 config 里的 input_dim，确保模型初始化正确
config['dataset']['input_dim'] = input_dim

model = MyFinalModel(config).to(DEVICE)

# === 4. 定义损失函数和优化器 ===
criterion_mse = nn.MSELoss()
criterion_cl = ContrastiveLoss(BATCH_SIZE, device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === 5. 开始训练 ===
print("\n=== 🚀 开始重训 (Retraining) ===")
model.train()
total_start_time = time.time()

best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_cl_loss = 0.0
    
    for i, batch in enumerate(train_loader):
        x = batch.to(DEVICE)
        
        # Target
        target_next = x[:, -1, :] # [B, N]
        target_window = x         # [B, W, N]

        optimizer.zero_grad()

        # A. 主任务 (Forward)
        pred_next, recon_window, _ = model(x)
        
        loss_forecast = criterion_mse(pred_next, target_next)
        loss_recon = criterion_mse(recon_window, target_window)
        loss_main = loss_forecast + loss_recon

        # B. 对比学习 (Forward Encoder Only)
        x_i = data_augmentation(x)
        x_j = data_augmentation(x)
        
        # 【修正点】这里必须调用 metric_encoder
        z_i = model.metric_encoder(x_i) 
        z_j = model.metric_encoder(x_j)
        
        # 展平做对比
        z_i_flat = z_i.view(BATCH_SIZE, -1)
        z_j_flat = z_j.view(BATCH_SIZE, -1)
        
        loss_cl = criterion_cl(z_i_flat, z_j_flat)

        # Total Loss
        loss_total = loss_main + (LAMBDA_CL * loss_cl)

        loss_total.backward()
        optimizer.step()

        epoch_loss += loss_total.item()
        epoch_cl_loss += loss_cl.item()

    # 打印进度
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # 早停逻辑
    if avg_loss < (best_loss - MIN_DELTA):
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"🛑 早停触发！最佳 Loss: {best_loss:.4f}")
            break

print(f"✅ 训练结束！新模型已保存至: {save_path}")