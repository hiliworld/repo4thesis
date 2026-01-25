import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import time
import copy
from step15_hdfs_dataset import get_hdfs_loaders
from model_v2_with_gat import MyFinalModel

# === 配置参数 ===
CONFIG_FILE = 'config.yaml'
BATCH_SIZE = 64
EPOCHS = 1000           # 设大一点，反正有早停
LR = 0.001
PATIENCE = 5          # 容忍度：如果连续 5 次没变好，就停
MODEL_SAVE_PATH = "hdfs_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 设备: {DEVICE} | 模式: LOG ANOMALY DETECTION (With Early Stopping)")

# 1. 加载配置和数据
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# 强制覆盖为 log 模式
config['dataset']['modality'] = 'log'

# 获取数据加载器
print("🔄 正在加载数据...")
train_loader, test_loader, vocab_size = get_hdfs_loaders(batch_size=BATCH_SIZE)
config['dataset']['vocab_size'] = vocab_size 
print(f"📚 词表大小: {vocab_size}")

# 2. 初始化模型
model = MyFinalModel(config).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 3. 辅助函数：计算验证集 Loss
def validate(model, loader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            input_seq = x[:, :-1]
            target_token = x[:, -1]
            
            logits, _ = model(input_seq)
            loss = criterion(logits, target_token)
            total_val_loss += loss.item()
    return total_val_loss / len(loader)

# 4. 训练循环 (带早停)
print("\n=== 🚀 开始训练 (Patience={}) ===".format(PATIENCE))

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    # --- 训练阶段 ---
    model.train()
    total_train_loss = 0
    start_time = time.time()
    
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(DEVICE)
        input_seq = batch_x[:, :-1]
        target_token = batch_x[:, -1]
        
        optimizer.zero_grad()
        logits, _ = model(input_seq)
        loss = criterion(logits, target_token)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # --- 验证阶段 ---
    avg_val_loss = validate(model, test_loader)
    
    epoch_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Time: {epoch_time:.1f}s | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
    
    # --- 早停检查 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0 # 重置计数器
        # 保存最佳模型权重到内存，或者直接写盘
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" 🌟 Best (Saved)")
    else:
        patience_counter += 1
        print(f" ⏳ Patience {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print(f"\n🛑 早停触发！在第 {epoch+1} 轮停止。最佳 Val Loss: {best_val_loss:.4f}")
        break

print(f"✅ 训练结束。最佳模型已保存至 {MODEL_SAVE_PATH}")