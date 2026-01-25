import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保能导入 dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import UACDataset
from model import UACModel

# === 配置 ===
CONFIG = {
    'data_dir': 'data/final_dataset',
    'metric_dim': 333,
    'metric_window': 20,
    'embed_dim': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint': 'checkpoints/uac_model_best.pth',
    'plot_limit': 3000  
}

def find_label_key(batch_keys):
    """自动寻找标签的键名"""
    candidates = ['label', 'labels', 'ground_truth', 'y', 'target']
    for key in candidates:
        if key in batch_keys:
            return key
    return None

def visualize():
    print(f"🎨 [Visualization] 启动诊断...")
    
    # 1. 准备依赖
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0]
        print(f"✅ 加载语义向量，词表大小: {vocab_size}")
    else:
        vocab_size = 3000
        weights = None
        print("⚠️ 使用默认词表")
    
    # 2. 初始化模型
    model = UACModel(metric_dim=CONFIG['metric_dim'], 
                     log_vocab_size=vocab_size,
                     embed_dim=CONFIG['embed_dim'],
                     log_weights=weights)
    
    # 加载权重
    if not os.path.exists(CONFIG['checkpoint']):
        print(f"❌ 找不到权重文件: {CONFIG['checkpoint']}")
        return

    checkpoint = torch.load(CONFIG['checkpoint'], map_location=CONFIG['device'])
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.to(CONFIG['device'])
    model.eval()
    
    # 3. 加载数据 (使用 test 模式)
    try:
        test_ds = UACDataset(CONFIG['data_dir'], mode='test', metric_window=CONFIG['metric_window'], vocab_size=vocab_size)
    except Exception as e:
        print(f"❌ 加载 Dataset 失败: {e}")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"✅ test 集加载完毕！样本数: {len(test_ds)}")
    
    # 4. 推理 & 收集
    scores = []
    labels = []
    label_key = None # 缓存键名
    
    print("   -> 正在推理...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            if i >= CONFIG['plot_limit']: break 
            
            # === 【核心修复】自动检测键名 ===
            if label_key is None:
                # 打印一次所有键名，方便调试
                print(f"\n🔍 [Debug] Batch Keys: {list(batch.keys())}")
                label_key = find_label_key(batch.keys())
                if label_key is None:
                    print("❌ 无法找到标签 Key！请检查 dataset.py 返回的字典键名。")
                    return
                print(f"✅ 锁定标签 Key: '{label_key}'")

            # 移动数据到 GPU
            m_seq = batch['metric_seq'].to(CONFIG['device'])
            m_mask = batch['metric_mask'].to(CONFIG['device'])
            l_seq = batch['log_seq'].to(CONFIG['device'])
            l_mask = batch['log_mask'].to(CONFIG['device'])
            l_count = batch['log_count'].to(CONFIG['device']) if 'log_count' in batch else None
            
            # 获取标签
            y = batch[label_key].item()
            
            # Forward
            p_m, p_l, _ = model(m_seq, m_mask, l_seq, l_mask, log_count=l_count)
            sim = F.cosine_similarity(p_m, p_l)
            score = 1 - sim.item()
            
            scores.append(score)
            labels.append(y)
            
    # 5. 绘图
    scores = np.array(scores)
    labels = np.array(labels)
    x = np.arange(len(scores))
    
    plt.figure(figsize=(15, 8))
    plt.plot(x, scores, label='Anomaly Score', color='dodgerblue', alpha=0.8)
    
    fault_indices = np.where(labels == 1)[0]
    if len(fault_indices) > 0:
        plt.scatter(fault_indices, scores[fault_indices], color='red', s=15, label='Actual Fault', zorder=5)
    else:
        print("⚠️ 前 N 个样本中未发现故障标签。")

    plt.title("Diagnostic Plot: Anomaly Scores vs Ground Truth")
    plt.xlabel("Time Steps")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "anomaly_diagnosis.png"
    plt.savefig(save_path)
    print(f"\n✅ 图片已保存: {save_path}")

if __name__ == "__main__":
    visualize()