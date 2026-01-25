import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import UACModel
from dataset import UACDataset

# === 配置 ===
BATCH_SIZE = 64
LR = 1e-3  # 用大一点的学习率，强行过拟合
STEPS = 300 # 跑 100 轮
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(parent_dir, 'data/final_dataset')

def contrastive_loss(m_feat, l_feat, temp_val=0.07):
    # 手写一个简单的 Loss 用于测试，排除 Learnable Loss 的干扰
    logits = torch.matmul(m_feat, l_feat.T) / temp_val
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss_m = torch.nn.functional.cross_entropy(logits, labels)
    loss_l = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_m + loss_l) / 2

def run_overfit_test():
    print(f"🧪 [Overfit Test] 正在尝试让模型死记硬背 1 个 Batch...")
    
    # 1. 初始化带 Projector 的新模型
    # 注意：这里假设你已经把 Model 改成了带 ProjectionHead 的版本
    # 为了测试严谨性，我们先用硬编码的 vocab_size
    model = UACModel(metric_dim=333, log_vocab_size=3000, embed_dim=128)
    model.to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 2. 只取一个 Batch 的数据，并固定住
    ds = UACDataset(DATA_DIR, mode='train', metric_window=20)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    try:
        fixed_batch = next(iter(loader))
    except:
        print("❌ 数据不足")
        return

    m_seq = fixed_batch['metric_seq'].to(DEVICE)
    m_mask = fixed_batch['metric_mask'].to(DEVICE)
    l_seq = fixed_batch['log_seq'].to(DEVICE)
    l_mask = fixed_batch['log_mask'].to(DEVICE)
    
    print(f"   -> 数据加载成功: Shape {m_seq.shape}")
    
    history = []
    sim_history = []

    # 3. 疯狂循环训练这同一个 Batch
    for step in range(STEPS):
        optimizer.zero_grad()
        
        # Forward
        m_out, l_out = model(m_seq, m_mask, l_seq, l_mask)
        
        # Loss
        loss = contrastive_loss(m_out, l_out)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 记录状态
        with torch.no_grad():
            # 计算当前正样本平均相似度
            sim_matrix = torch.matmul(m_out, l_out.T)
            pos_sim = torch.diag(sim_matrix).mean().item()
            
        history.append(loss.item())
        sim_history.append(pos_sim)
        
        if step % 10 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Similarity: {pos_sim:.4f}")

    # 4. 结果判定
    final_sim = sim_history[-1]
    print("\n" + "="*40)
    print(f"🏁 测试结束 | 最终相似度: {final_sim:.4f}")
    
    if final_sim > 0.9:
        print("✅ [成功] 模型具备学习能力！")
        print("   -> 结论：代码逻辑无误，加上 Projector 后肯定能学好。")
        print("   -> 建议：立刻去跑全量训练 (train.py)。")
    elif final_sim > 0.6:
        print("⚠️ [一般] 能学，但有点费劲。")
        print("   -> 结论：可能是 Embedding 维度太小或数据噪音太大。")
    else:
        print("❌ [失败] 连 1 个 Batch 都背不下来 (Sim < 0.6)。")
        print("   -> 结论：代码有硬伤！可能是 Mask 写反了，或者 Encoder 输出全是 0。")
        print("   -> 此时不要去跑 train.py，那是浪费时间。")

    # 画图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history, label='Loss')
    plt.title('Loss Curve (Overfit)')
    plt.subplot(1, 2, 2)
    plt.plot(sim_history, color='orange', label='Similarity')
    plt.title('Positive Similarity (Should -> 1.0)')
    plt.savefig('overfit_test_result.png')
    print("📸 结果图: overfit_test_result.png")

if __name__ == "__main__":
    run_overfit_test()