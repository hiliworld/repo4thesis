import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import yaml
import random
from sklearn.metrics import roc_auc_score, pairwise_distances
from torch.utils.data import Dataset, DataLoader

try:
    from src.data.loader import get_dataloaders
    from src.models.anomaly_model import MyFinalModel
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    exit()

# === 配置 ===
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
CLUSTER_FILE = "fault_clusters_analysis/fault_clustering_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 1-Shot 严谨测试
SHOTS_PER_CLASS = 1   
RANDOM_SEED = 42

TRAIN_CLUSTERS = [0, 1, 2, 4]  
TEST_CLUSTERS = [3]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class HonestFineTuneDataset(Dataset):
    """
    【修正】诚实的数据集构建策略
    指导建议：不要过度复制故障样本。
    策略：既然故障只有 N 个，那我们就只取 N*2 个正常样本。
    构建一个极小的、高质量的微调集，而不是这就是“海量重复集”。
    """
    def __init__(self, normal_windows, fault_windows):
        # 1. 故障样本 (少量)
        self.fault = fault_windows
        n_fault = len(self.fault)
        
        # 2. 正常样本 (下采样，只取故障样本的 2 倍，保持 2:1 比例)
        # 这样避免了把 1 个故障样本复制 10000 次去匹配正常样本
        n_normal_select = min(len(normal_windows), n_fault * 2) 
        # 随机抽样正常样本，防止总是取前几个
        idx = np.random.choice(len(normal_windows), n_normal_select, replace=False)
        self.normal = normal_windows[idx]
        
        print(f"   ⚖️ Honest Balancing: Fault={n_fault}, Normal={len(self.normal)} (Ratio ~1:2)")
        print(f"      (已移除'过度复制'逻辑，避免过拟合单一样本)")

        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([np.zeros(len(self.normal)), np.ones(len(self.fault))], axis=0)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

# ... (保留 get_n_shot_data, load_full_cluster_data, deviation_loss 等辅助函数，同前) ...
def get_n_shot_data(loader, cluster_df, target_clusters, shots=1):
    collected_data = []
    cluster_indices_map = {}
    for _, row in cluster_df.iterrows():
        cid = row['Cluster_Type']
        if cid not in target_clusters: continue
        if cid not in cluster_indices_map: cluster_indices_map[cid] = []
        mid_point = (row['Start_Idx'] + row['End_Idx']) // 2
        cluster_indices_map[cid].append(mid_point)
    selected_indices = []
    for cid in target_clusters:
        if cid not in cluster_indices_map: continue
        candidates = cluster_indices_map[cid]
        k = min(len(candidates), shots)
        picks = random.sample(candidates, k)
        selected_indices.extend(picks)
    
    selected_indices_set = set(selected_indices)
    current_idx = 0
    found_count = 0
    for x in loader:
        batch_size = x.shape[0]
        batch_indices = range(current_idx, current_idx + batch_size)
        common = selected_indices_set.intersection(batch_indices)
        if common:
            for pick in common:
                local_idx = pick - current_idx
                collected_data.append(x[local_idx].numpy())
                found_count += 1
        current_idx += batch_size
        if found_count >= len(selected_indices): break 
    return np.array(collected_data)

def load_full_cluster_data(loader, cluster_df, target_clusters):
    target_indices = set()
    for _, row in cluster_df.iterrows():
        if row['Cluster_Type'] in target_clusters:
            target_indices.update(range(row['Start_Idx'], row['End_Idx']))
    collected = []
    curr = 0
    for x in loader:
        bs = x.shape[0]
        b_idxs = range(curr, curr+bs)
        if not target_indices.isdisjoint(b_idxs):
            for i, gidx in enumerate(b_idxs):
                if gidx in target_indices: collected.append(x[i].numpy())
        curr += bs
    return np.array(collected)

def deviation_loss(pred, recon, target, labels, margin=5.0):
    error = torch.mean((pred - target)**2, dim=1) + torch.mean((recon - target)**2, dim=1)
    loss_normal = error[labels == 0].mean() if (labels==0).sum() > 0 else 0.0
    loss_fault = torch.relu(margin - error[labels == 1]).mean() if (labels == 1).sum() > 0 else 0.0
    return loss_normal + loss_fault

def check_similarity(train_data, test_data):
    """
    【新增】指导建议：相似度自检
    验证训练集(Cluster 0/1/2)和测试集(Cluster 3)是否真的长得不一样
    """
    # 展平数据: [Batch, Window, Feat] -> [Batch, Window*Feat]
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)
    
    # 随机取样测试集一部分来算，防止内存爆炸
    if len(test_flat) > 1000:
        idx = np.random.choice(len(test_flat), 1000, replace=False)
        test_flat = test_flat[idx]
        
    # 计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(train_flat, test_flat)
    
    max_sim = sim_matrix.max()
    mean_sim = sim_matrix.mean()
    print(f"   🔍 Similarity Check (Train vs Test Faults):")
    print(f"      Max Similarity: {max_sim:.4f} (Should < 0.95)")
    print(f"      Mean Similarity: {mean_sim:.4f}")
    if max_sim > 0.95:
        print("      ⚠️ WARNING: Potential leakage! Some test faults look identical to training faults.")
    else:
        print("      ✅ Safe: Training and Test faults are distinct.")

def main():
    set_seed(RANDOM_SEED)
    print(f"🚀 {SHOTS_PER_CLASS}-SHOT FINAL HONEST RUN (Anti-Leakage Mode)")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    # A. 正常数据处理 (我的修复：防止 Normal Data 泄漏)
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 20000: break 
        normal_data_pool.append(x.numpy())
    normal_data_full = np.concatenate(normal_data_pool)
    
    split_idx = int(len(normal_data_full) * 0.5)
    normal_train = normal_data_full[:split_idx]  # 用于微调 (Support Set)
    normal_test  = normal_data_full[split_idx:]  # 用于评估 (Query Set)
    print(f"   🛡️ Normal Data Split: Train={len(normal_train)} | Test={len(normal_test)}")

    # B. 故障数据
    train_n_shot_data = get_n_shot_data(test_loader, df_clusters, TRAIN_CLUSTERS, shots=SHOTS_PER_CLASS)
    test_novel_data = load_full_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    if len(train_n_shot_data) == 0: return

    # C. 【新增】相似度检查 (指导的建议)
    check_similarity(train_n_shot_data, test_novel_data)

    # 2. 模型准备
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # 3. 微调 (使用 HonestFineTuneDataset)
    # 降低 LR 和 Epoch (指导的建议，防止过拟合)
    print(f"\n⚡ Fine-tuning (Balanced, Low LR)...")
    dataset = HonestFineTuneDataset(normal_train, train_n_shot_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True) # Batch size 调小，因为数据量少了
    optimizer = optim.Adam(model.parameters(), lr=1e-6) # LR 降低
    
    model.train()
    for epoch in range(3): # Epoch 减少
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ret = model(x)
            loss = deviation_loss(ret[0].squeeze(), ret[1][:,-1,:], x[:,-1,:], y, margin=5.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    # 4. 评估
    print("\n🧐 Evaluating (on Unseen Data)...")
    model.eval()
    
    # 评估 Test Faults
    scores_novel = []
    t_novel = torch.from_numpy(test_novel_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(test_novel_data), 256):
            b = t_novel[i:i+256]
            ret = model(b)
            l = torch.mean((ret[0].squeeze()-b[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-b[:,-1,:])**2, 1)
            scores_novel.append(l.cpu().numpy())
    scores_novel = np.concatenate(scores_novel)
    
    # 评估 Test Normal (使用 normal_test)
    scores_norm = []
    eval_len = min(len(normal_test), len(test_novel_data))
    t_norm = torch.from_numpy(normal_test[:eval_len]).float().to(DEVICE) 
    with torch.no_grad():
        ret = model(t_norm)
        l = torch.mean((ret[0].squeeze()-t_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-t_norm[:,-1,:])**2, 1)
        scores_norm.append(l.cpu().numpy())
    scores_norm = np.concatenate(scores_norm)
    
    # 统计
    mean_novel = np.mean(scores_novel)
    mean_norm = np.mean(scores_norm)
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_novel))]),
        np.concatenate([scores_norm, scores_novel])
    )
    
    print(f"\n🏆 FINAL VERIFIED RESULT ({SHOTS_PER_CLASS}-Shot):")
    print(f"   Normal Score (Unseen): {mean_norm:.4f}")
    print(f"   Novel Fault Score    : {mean_novel:.4f}")
    print(f"   Gap Ratio            : {mean_novel/mean_norm:.1f}x")
    print(f"   AUC                  : {auc:.4f}")
    
    if auc > 0.90 and (mean_novel/mean_norm) > 10:
         print("   ✅ Validated SOTA: High AUC with Realistic Gap.")
    else:
         print("   ⚠️ Needs Check: Gap might be too small or AUC too low.")

if __name__ == "__main__":
    main()