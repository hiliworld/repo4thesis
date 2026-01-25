import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch.utils.data import DataLoader

# 引入你的组件
from dataset import UACDataset
from model import UACModel

# === 配置 ===
CHECKPOINT_PATH = 'checkpoints/uac_model_best.pth' # 加载你现在那个 2.36 loss 的模型
DATA_DIR = 'data/final_dataset'
BATCH_SIZE = 64 # 诊断时不用太大
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def diagnose():
    print(f"🩺 [Diagnosis] 正在对模型进行解剖... Device: {DEVICE}")

    # 1. 加载模型
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ 找不到模型文件，请确认路径！")
        return
        
    print("📥 加载模型权重...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # 自动适配权重格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # 获取词表大小 (从权重推断，避免硬编码)
    if 'log_encoder.embedding.weight' in state_dict:
        vocab_size = state_dict['log_encoder.embedding.weight'].shape[0]
    else:
        vocab_size = 1000 # Fallback
        
    model = UACModel(metric_dim=333, log_vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 2. 加载一小批数据 (测试集)
    test_ds = UACDataset(DATA_DIR, mode='test', metric_window=20)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 取一个 Batch 出来分析
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("❌ 数据集为空！")
        return

    m_seq = batch['metric_seq'].to(DEVICE)
    m_mask = batch['metric_mask'].to(DEVICE)
    l_seq = batch['log_seq'].to(DEVICE)
    l_mask = batch['log_mask'].to(DEVICE)

    with torch.no_grad():
        # 获取特征向量
        feat_m, feat_l = model(m_seq, m_mask, l_seq, l_mask)
        # feat_m, feat_l 已经是 normalize 过的了，所以点积就是余弦相似度
        
        # 计算相似度矩阵 [B, B]
        # sim_matrix[i][j] = Metric[i] 和 Log[j] 的相似度
        sim_matrix = torch.matmul(feat_m, feat_l.T).cpu().numpy()

    # === 诊断图 1: 相似度矩阵热力图 ===
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Similarity Matrix (Batch 64x64)\nIdeally: Diagonal is Red, Others Blue")
    plt.xlabel("Log Index")
    plt.ylabel("Metric Index")

    # === 诊断图 2: 正负样本分布直方图 ===
    # 正样本：对角线元素 (Diagonal)
    pos_sims = np.diag(sim_matrix)
    # 负样本：非对角线元素 (Off-diagonal)
    mask = ~np.eye(BATCH_SIZE, dtype=bool)
    neg_sims = sim_matrix[mask]

    plt.subplot(1, 3, 2)
    sns.kdeplot(pos_sims, fill=True, color='red', label='Positive Pairs (Matched)')
    sns.kdeplot(neg_sims, fill=True, color='blue', label='Negative Pairs (Unmatched)')
    plt.title(f"Similarity Distribution\nPos Avg: {pos_sims.mean():.3f} | Neg Avg: {neg_sims.mean():.3f}")
    plt.xlabel("Cosine Similarity (-1 to 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # === 诊断图 3: 模拟异常检测得分 ===
    # 假设我们按顺序取一段数据，看看 Score 的波动
    # 正常分应该低 (Sim高)，异常分应该高 (Sim低)
    anomaly_scores = 1 - pos_sims # Score = 1 - CosSim
    
    plt.subplot(1, 3, 3)
    plt.plot(anomaly_scores, marker='o', linestyle='-', color='purple')
    plt.title("Anomaly Scores in this Batch\n(Should vary, not flat)")
    plt.ylim(0, 2) # Cosine Distance 范围是 0~2
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score (1 - Sim)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('diagnosis_report.png')
    print("✅ 诊断完成！请查看生成的图片: diagnosis_report.png")
    
    # === 打印关键数值 ===
    print("\n" + "="*30)
    print("📊 关键数值分析报告")
    print("="*30)
    print(f"1. 正样本平均相似度 (越高越好): {pos_sims.mean():.4f}")
    print(f"2. 负样本平均相似度 (越低越好): {neg_sims.mean():.4f}")
    print(f"3. 区分度 (Delta): {pos_sims.mean() - neg_sims.mean():.4f}")
    print("-" * 30)
    
    diff = pos_sims.mean() - neg_sims.mean()
    if diff < 0.1:
        print("🚨 [严重警告] 模型发生 '坍塌 (Collapse)'！")
        print("   Metric 和 Log 向量几乎是随机的，或者全都映射到了同一个点。")
        print("   原因推测：学习率太大导致震荡，或 log_encoder 没学到东西。")
    elif diff < 0.3:
        print("⚠️ [警告] 模型区分度很低。")
        print("   它能分清一点点，但很容易混淆。")
    else:
        print("✅ [正常] 模型具有一定的区分能力。")
    print("="*30)

if __name__ == "__main__":
    diagnose()