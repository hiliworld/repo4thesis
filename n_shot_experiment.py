import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import yaml
import random
from sklearn.metrics import roc_auc_score
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

# 🔥 1-Shot 设置
SHOTS_PER_CLASS = 5  
RANDOM_SEED = 42
#case 1:
#TRAIN_CLUSTERS = [0, 1, 2, 4]  
#TEST_CLUSTERS = [3]
#TRAIN_CLUSTERS = [0, 2, 3, 4] 
#TEST_CLUSTERS = [1]
#TRAIN_CLUSTERS = [0, 1, 3, 4] 
#TEST_CLUSTERS = [2]
TRAIN_CLUSTERS = [2, 1, 3, 4] 
TEST_CLUSTERS = [0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HonestFineTuneDataset(Dataset):
    """诚实平衡数据集"""
    def __init__(self, normal_windows, fault_windows):
        self.fault = fault_windows
        n_fault = len(self.fault)
        n_normal_select = min(len(normal_windows), n_fault * 2) 
        idx = np.random.choice(len(normal_windows), n_normal_select, replace=False)
        self.normal = normal_windows[idx]
        print(f"   ⚖️ Honest Balancing: Fault={n_fault}, Normal={len(self.normal)} (Ratio ~1:2)")
        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([np.zeros(len(self.normal)), np.ones(len(self.fault))], axis=0)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

# 辅助函数
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

def evaluate_model(model, normal_data, fault_data, batch_size=256):
    """通用的评估函数"""
    model.eval()
    
    # Fault Scores
    scores_fault = []
    t_fault = torch.from_numpy(fault_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(fault_data), batch_size):
            b = t_fault[i:i+batch_size]
            ret = model(b)
            l = torch.mean((ret[0].squeeze()-b[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-b[:,-1,:])**2, 1)
            scores_fault.append(l.cpu().numpy())
    scores_fault = np.concatenate(scores_fault)
    
    # Normal Scores
    scores_norm = []
    eval_len = min(len(normal_data), len(fault_data)) # 平衡对比
    t_norm = torch.from_numpy(normal_data[:eval_len]).float().to(DEVICE) 
    with torch.no_grad():
        ret = model(t_norm)
        l = torch.mean((ret[0].squeeze()-t_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-t_norm[:,-1,:])**2, 1)
        scores_norm.append(l.cpu().numpy())
    scores_norm = np.concatenate(scores_norm)
    
    mean_fault = np.mean(scores_fault)
    mean_norm = np.mean(scores_norm)
    gap = mean_fault / mean_norm if mean_norm > 1e-9 else 0.0
    
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_fault))]),
        np.concatenate([scores_norm, scores_fault])
    )
    
    return mean_norm, mean_fault, gap, auc

def main():
    set_seed(RANDOM_SEED)
    print(f"🚀 {SHOTS_PER_CLASS}-SHOT COMPARISON EXPERIMENT (Zero-Shot vs Few-Shot)")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    # A. 正常数据处理
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 20000: break 
        normal_data_pool.append(x.numpy())
    normal_data_full = np.concatenate(normal_data_pool)
    
    split_idx = int(len(normal_data_full) * 0.5)
    normal_train = normal_data_full[:split_idx] 
    normal_test  = normal_data_full[split_idx:] 

    # B. 故障数据
    train_n_shot_data = get_n_shot_data(test_loader, df_clusters, TRAIN_CLUSTERS, shots=SHOTS_PER_CLASS)
    test_novel_data = load_full_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    if len(train_n_shot_data) == 0: return

    # 2. 模型准备
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # === STEP 0: 微调前评估 (Baseline) ===
    print("\n📊 Step 0: Evaluating Baseline (Zero-Shot)...")
    base_norm, base_fault, base_gap, base_auc = evaluate_model(model, normal_test, test_novel_data)
    print(f"   [Baseline] Gap: {base_gap:.1f}x | AUC: {base_auc:.4f} | Fault Score: {base_fault:.4f}")

    # === STEP 1: 微调 ===
    print(f"\n⚡ Step 1: Fine-tuning ({SHOTS_PER_CLASS}-Shot)...")
    dataset = HonestFineTuneDataset(normal_train, train_n_shot_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    
    model.train()
    for epoch in range(3):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ret = model(x)
            loss = deviation_loss(ret[0].squeeze(), ret[1][:,-1,:], x[:,-1,:], y, margin=5.0)
            loss.backward()
            optimizer.step()
        
    # === STEP 2: 微调后评估 (Ours) ===
    print("\n📊 Step 2: Evaluating Fine-tuned Model (Few-Shot)...")
    ft_norm, ft_fault, ft_gap, ft_auc = evaluate_model(model, normal_test, test_novel_data)
    
    # === 最终对比报告 ===
    print("\n" + "="*50)
    print("🏆 IMPACT ANALYSIS REPORT")
    print("="*50)
    print(f"{'Metric':<15} | {'Baseline (Zero-Shot)':<20} | {'Ours (1-Shot)':<15} | {'Improvement'}")
    print("-" * 65)
    print(f"{'Fault Score':<15} | {base_fault:<20.4f} | {ft_fault:<15.4f} | {ft_fault/base_fault:.1f}x 🚀")
    print(f"{'Normal Score':<15} | {base_norm:<20.4f} | {ft_norm:<15.4f} | (Stable)")
    print(f"{'Gap Ratio':<15} | {base_gap:<20.1f}x             | {ft_gap:<15.1f}x      | +{ft_gap - base_gap:.1f}x")
    print(f"{'AUC':<15} | {base_auc:<20.4f} | {ft_auc:<15.4f} | +{(ft_auc-base_auc)*100:.2f}%")
    print("="*50)
    
    if ft_gap > base_gap * 2:
        print("✅ Conclusion: Few-shot learning significantly amplified the anomaly signal.")
    else:
        print("⚠️ Conclusion: Improvement is marginal.")

if __name__ == "__main__":
    main()