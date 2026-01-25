import sys
import os

# === 核心修复：把父目录加入系统路径 ===
# 获取当前脚本所在目录 (/home/sde/MyThesis/aiops_test_unit)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (/home/sde/MyThesis)
parent_dir = os.path.dirname(current_dir)
# 把父目录加入 Python 搜索路径，这样就能找到 dataset.py 和 model.py 了
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

# 现在可以正常导入了
from dataset import UACDataset
from model import UACModel

# === 配置 (自动适配路径) ===
# 使用绝对路径，确保无论在哪运行都能找到文件
CHECKPOINT_PATH = os.path.join(parent_dir, 'checkpoints/uac_model_best.pth')
DATA_DIR = os.path.join(parent_dir, 'data/final_dataset')

BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def diagnose():
    print(f"🩺 [Diagnosis V4] 正在检查正样本对齐情况... Device: {DEVICE}")
    print(f"   项目根目录: {parent_dir}")

    # 1. 加载模型
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 找不到模型权重: {CHECKPOINT_PATH}")
        print("   -> 请确认 train.py 已经跑完至少一个 Epoch 并保存了模型。")
        return
        
    print(f"📥 加载权重: {os.path.basename(CHECKPOINT_PATH)}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # 自动适配权重字典
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # 顺便看看现在的温度是多少
        if 'temp_param' in checkpoint:
            # 提取 logit_scale
            try:
                logit_scale = checkpoint['temp_param'].get('logit_scale', torch.tensor(0.0))
                temp_val = 1.0 / logit_scale.exp().item() # 温度 = 1/exp(scale)
                print(f"🔥 当前模型学习到的温度 (Temperature): {temp_val:.4f}")
            except:
                pass
    else:
        state_dict = checkpoint
        
    # 获取词表大小
    if 'log_encoder.embedding.weight' in state_dict:
        vocab_size = state_dict['log_encoder.embedding.weight'].shape[0]
    else:
        vocab_size = 1000
        
    model = UACModel(metric_dim=333, log_vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 2. 加载测试数据
    # 注意：这里需要确保 Dataset 类能处理正确的路径
    # 如果 UACDataset 内部拼接路径有问题，可能需要把 absolute path 传进去
    # 这里假设 UACDataset 接收的是 DATA_DIR
    test_ds = UACDataset(DATA_DIR, mode='test', metric_window=20)
    
    # 防止数据量不够 BatchSize 报错
    drop_last = len(test_ds) > BATCH_SIZE
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("❌ 数据集为空或不足一个 Batch")
        return

    m_seq = batch['metric_seq'].to(DEVICE)
    m_mask = batch['metric_mask'].to(DEVICE)
    l_seq = batch['log_seq'].to(DEVICE)
    l_mask = batch['log_mask'].to(DEVICE)

    with torch.no_grad():
        # 获取特征向量
        feat_m, feat_l = model(m_seq, m_mask, l_seq, l_mask)
        
        # 计算相似度矩阵 [B, B]
        sim_matrix = torch.matmul(feat_m, feat_l.T).cpu().numpy()

    # === 核心指标计算 ===
    # 正样本：对角线
    pos_sims = np.diag(sim_matrix)
    # 负样本：非对角线
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    neg_sims = sim_matrix[mask]

    print("\n" + "="*40)
    print("📊 [对齐性诊断报告]")
    print("="*40)
    print(f"1. 正样本相似度 (Alignment): {pos_sims.mean():.4f} (目标: > 0.6)")
    print(f"2. 负样本相似度 (Uniformity): {neg_sims.mean():.4f} (目标: < 0.1)")
    print(f"3. 区分度 (Delta):           {pos_sims.mean() - neg_sims.mean():.4f} (越大越好)")
    print("-" * 40)

    # === 绘图 ===
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pos_sims, fill=True, color='red', label='Positive Pairs (Matched)')
    try:
        sns.kdeplot(neg_sims, fill=True, color='blue', label='Negative Pairs (Unmatched)')
    except:
        pass # 防止负样本太少报错
        
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Similarity Distribution (Batch={len(pos_sims)})\nDelta={pos_sims.mean() - neg_sims.mean():.3f}")
    plt.xlabel("Cosine Similarity (-1 to 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(current_dir, 'diagnosis_v4_alignment.png')
    plt.savefig(save_path)
    print(f"📸 诊断图已保存: {save_path}")

if __name__ == "__main__":
    diagnose()