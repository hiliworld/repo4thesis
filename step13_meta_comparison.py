import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
from step12_meta_dataloader import MetaTaskSampler
from model_v2_with_gat import MyFinalModel

# === 配置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 100
FEATURE_DIM = 36
K_SHOT = 10         
FINE_TUNE_STEPS = 50  # 稍微增加一点步数，让差异更明显
LR = 0.001            # Adam 用 0.001 是黄金学习率

# === 1. 准备数据 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(base_dir, 'data', 'ServerMachineDataset', 'test')

# 抽取任务
sampler = MetaTaskSampler(test_data_dir, window_size=WINDOW_SIZE, k_shot=K_SHOT, q_query=50)
print("🎲 正在抽取一个从未见过的测试任务...")
supports, queries = sampler.get_batch(meta_batch_size=1) 

x_support = supports[0].to(DEVICE)
x_query = queries[0].to(DEVICE)

# ✅ 检查点：确认数据是否归一化
print(f"📊 数据检查 | Support Max: {x_support.max():.4f}, Min: {x_support.min():.4f}")
if x_support.max() > 10.0:
    print("⚠️ 警告：数据似乎未归一化！Loss 可能会爆炸。请检查 step12_meta_dataloader.py")

# === 2. 准备两个选手 ===
print("🥊 选手就位...")

# 选手 A: 普通预训练模型 (Baseline)
model_baseline = MyFinalModel(num_features=FEATURE_DIM, window_size=WINDOW_SIZE).to(DEVICE)
baseline_path = "my_trained_model_adaptive.pth"
if os.path.exists(baseline_path):
    model_baseline.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
else:
    print(f"❌ 错误：找不到 {baseline_path}")

# 选手 B: 元学习模型 (Ours)
model_meta = MyFinalModel(num_features=FEATURE_DIM, window_size=WINDOW_SIZE).to(DEVICE)
meta_path = "my_meta_model_reptile.pth"
if os.path.exists(meta_path):
    model_meta.load_state_dict(torch.load(meta_path, map_location=DEVICE))
else:
    print(f"❌ 错误：找不到 {meta_path}")

criterion = nn.MSELoss()

# === 3. 定义适应过程 (Fine-tuning Loop) ===
def adapt_and_evaluate(model, name):
    # 深拷贝模型
    model_copy = copy.deepcopy(model)
    
    # ✅ 修改 1: 使用 Adam 优化器 (对 Baseline 更公平，收敛更快)
    optimizer = optim.Adam(model_copy.parameters(), lr=LR)
    
    losses = []
    
    print(f"   🏃 {name} 开始适应...")
    for step in range(FINE_TUNE_STEPS):
        # 1. 在 Support Set 上训练
        model_copy.train()
        p, r, _ = model_copy(x_support)
        loss = criterion(p, x_support[:, -1, :]) + criterion(r, x_support)
        
        optimizer.zero_grad()
        loss.backward()
        
        # ✅ 修改 2: 增加梯度裁剪 (防止 Loss 爆炸)
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 2. 在 Query Set 上评估
        model_copy.eval()
        with torch.no_grad():
            q_p, q_r, _ = model_copy(x_query)
            q_loss = criterion(q_p, x_query[:, -1, :]) + criterion(q_r, x_query)
            losses.append(q_loss.item())
            
    return losses

# === 4. 比赛开始 ===
losses_baseline = adapt_and_evaluate(model_baseline, "Baseline (Step 9)")
losses_meta = adapt_and_evaluate(model_meta, "Meta-Learning (Step 12)")

# === 5. 画图定胜负 ===
plt.figure(figsize=(10, 6))
# Baseline 用灰色虚线
plt.plot(losses_baseline, label='Baseline (Pre-trained)', linestyle='--', marker='o', color='gray', alpha=0.7)
# Meta 用红色实线，加粗
plt.plot(losses_meta, label='Ours (Meta-Learned)', linewidth=3, marker='s', color='red')

plt.title(f'Few-Shot Adaptation Comparison (K={K_SHOT})')
plt.xlabel('Gradient Steps (Adam Fine-tuning)')
plt.ylabel('Loss on New Machine (Query Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("meta_learning_comparison.png", dpi=300)
plt.show()

print("\n🏆 结果已生成！请查看 meta_learning_comparison.png")
print(f"Baseline 初始 Loss: {losses_baseline[0]:.4f} -> 最终: {losses_baseline[-1]:.4f}")
print(f"Meta     初始 Loss: {losses_meta[0]:.4f} -> 最终: {losses_meta[-1]:.4f}")