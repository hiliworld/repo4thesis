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

# ==========================================
# 🎛️ 实验配置 (基于 Ground Truth)
# ==========================================
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
# 确保这里指向的是 generate_ground_truth.py 生成的新文件
CLUSTER_FILE = "fault_clusters_analysis/fault_ground_truth.csv" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🎯 实验目标：跨维度泛化 (Cross-Dimension Generalization)
# 基类 (Train): 常见故障 (ID 0-4)
# 新类 (Test) : 全新维度的单点故障 (ID 5)
TRAIN_CLUSTERS = [0, 1, 2, 3, 4] 
TEST_CLUSTERS = [5]             

SHOTS_PER_CLASS = 1   # 1-Shot 挑战
RANDOM_SEED = 42

# ==========================================
# 🛠️ 工具类与函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AnchoredFineTuneDataset(Dataset):
    """
    ⚓ 锚点数据集策略：
    为了防止模型在学 1 个故障样本时'忘掉'什么是正常，
    我们需要大量引入正常样本作为'锚点' (Anchor)。
    比例推荐 1:10 (故障:正常)。
    """
    def __init__(self, normal_windows, fault_windows):
        self.fault = fault_windows
        n_fault = len(self.fault)
        
        # 策略：取 10 倍的正常样本
        n_normal_select = min(len(normal_windows), max(20, n_fault * 10))
        
        # 随机抽样
        idx = np.random.choice(len(normal_windows), n_normal_select, replace=False)
        self.normal = normal_windows[idx]
        
        print(f"   ⚓ Anchored Balancing: Fault={n_fault}, Normal={len(self.normal)} (Ratio ~1:{len(self.normal)//n_fault})")
        
        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([np.zeros(len(self.normal)), np.ones(len(self.fault))], axis=0)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

def get_n_shot_data(loader, cluster_df, target_clusters, shots=1):
    collected_data = []
    # 建立 Cluster -> [Event Indices] 的映射
    cluster_indices_map = {}
    for _, row in cluster_df.iterrows():
        cid = row['Cluster_Type']
        if cid not in target_clusters: continue
        if cid not in cluster_indices_map: cluster_indices_map[cid] = []
        # 取故障事件的中心点，最典型
        mid_point = (row['Start_Idx'] + row['End_Idx']) // 2
        cluster_indices_map[cid].append(mid_point)

    # 随机采样 N-Shot
    selected_indices = []
    for cid in target_clusters:
        if cid in cluster_indices_map:
            candidates = cluster_indices_map[cid]
            if not candidates: continue
            k = min(len(candidates), shots)
            picks = random.sample(candidates, k)
            selected_indices.extend(picks)
    
    # 从 DataLoader 提取数据
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
    """加载目标 Cluster 的全量数据用于评估"""
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
    model.eval()
    if len(fault_data) == 0: return 0, 0, 0, 0
    
    # 1. 计算故障分数
    scores_fault = []
    t_fault = torch.from_numpy(fault_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(fault_data), batch_size):
            b = t_fault[i:i+batch_size]
            ret = model(b)
            l = torch.mean((ret[0].squeeze()-b[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-b[:,-1,:])**2, 1)
            scores_fault.append(l.cpu().numpy())
    scores_fault = np.concatenate(scores_fault)
    
    # 2. 计算正常分数 (采样相同数量，保持公平)
    scores_norm = []
    eval_len = min(len(normal_data), len(fault_data))
    t_norm = torch.from_numpy(normal_data[:eval_len]).float().to(DEVICE)
    with torch.no_grad():
        ret = model(t_norm)
        l = torch.mean((ret[0].squeeze()-t_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-t_norm[:,-1,:])**2, 1)
        scores_norm.append(l.cpu().numpy())
    scores_norm = np.concatenate(scores_norm)
    
    # 3. 统计指标
    mean_fault = np.mean(scores_fault)
    mean_norm = np.mean(scores_norm)
    gap = mean_fault / mean_norm if mean_norm > 1e-9 else 0.0
    
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_fault))]),
        np.concatenate([scores_norm, scores_fault])
    )
    return mean_norm, mean_fault, gap, auc

# ==========================================
# 🚀 主程序
# ==========================================
def main():
    set_seed(RANDOM_SEED)
    print(f"🚀 1-SHOT ROBUST EXPERIMENT (Target: Cluster {TEST_CLUSTERS})")
    print("   Strategy: Freeze Backbone + Full-Batch Fine-Tuning")
    
    # 1. 加载数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    if not os.path.exists(CLUSTER_FILE):
        print(f"❌ 没找到 {CLUSTER_FILE}，请先运行 generate_ground_truth.py")
        return
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    # 准备正常数据 (Split Half)
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 20000: break 
        normal_data_pool.append(x.numpy())
    normal_data_full = np.concatenate(normal_data_pool)
    split_idx = int(len(normal_data_full) * 0.5)
    normal_train = normal_data_full[:split_idx] 
    normal_test  = normal_data_full[split_idx:] 

    # 准备故障数据
    train_n_shot_data = get_n_shot_data(test_loader, df_clusters, TRAIN_CLUSTERS, shots=SHOTS_PER_CLASS)
    test_novel_data = load_full_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    if len(train_n_shot_data) == 0:
        print("❌ Error: No training data (N-Shot) found.")
        return
    print(f"   Training Data (Support): {len(train_n_shot_data)} samples")
    print(f"   Testing Data (Query)   : {len(test_novel_data)} samples")

    # 2. 加载模型
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # === STEP 0: Baseline Check ===
    print("\n📊 Baseline (Zero-Shot) Evaluation...")
    b_norm, b_fault, b_gap, b_auc = evaluate_model(model, normal_test, test_novel_data)
    print(f"   [Base] Gap: {b_gap:.1f}x | AUC: {b_auc:.4f} | Fault: {b_fault:.4f}")

    # === STEP 1: Freeze Backbone (核心修改) ===
    print("\n❄️ Applying Freeze Strategy...")
    # 冻结特征提取器 (Encoder + GAT)，保护预训练知识不被破坏
    # 只解冻最后一层预测头 (假设层名包含 'head' 或 'pred' 或不在 encoder/gat 中)
    for name, param in model.named_parameters():
        if 'encoder' in name or 'gat' in name or 'feature' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # 只训练 head / predictor
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"   Backbone frozen. Tuning {len(trainable_params)} tensor groups only.")

    # === STEP 2: Full-Batch Fine-Tuning (核心修改) ===
    print(f"⚡ Fine-tuning ({SHOTS_PER_CLASS}-Shot)...")
    dataset = AnchoredFineTuneDataset(normal_train, train_n_shot_data)
    
    # 关键：使用 Full Batch (一次塞入所有数据)，消除 1-shot 的梯度随机性
    full_batch_size = len(dataset)
    train_loader = DataLoader(dataset, batch_size=full_batch_size, shuffle=True)
    
    optimizer = optim.Adam(trainable_params, lr=1e-3) # 只调一层，LR 可以稍微大点
    
    model.train()
    for epoch in range(10): 
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ret = model(x)
            # 关键：Margin 降为 1.0，防止分数过分膨胀
            loss = deviation_loss(ret[0].squeeze(), ret[1][:,-1,:], x[:,-1,:], y, margin=1.0)
            loss.backward()
            optimizer.step()
        
    # === STEP 3: Evaluation ===
    print("\n📊 Final (Few-Shot) Evaluation...")
    f_norm, f_fault, f_gap, f_auc = evaluate_model(model, normal_test, test_novel_data)
    
    # === 最终报告 ===
    print("\n" + "="*50)
    print(f"🏆 FINAL RESULT REPORT (Cluster {TEST_CLUSTERS})")
    print("="*50)
    print(f"{'Metric':<15} | {'Baseline (Zero-Shot)':<20} | {'Ours (Frozen)':<15} | {'Improvement'}")
    print("-" * 65)
    print(f"{'Fault Score':<15} | {b_fault:<20.4f} | {f_fault:<15.4f} | {f_fault/b_fault:.2f}x 🚀")
    print(f"{'Normal Score':<15} | {b_norm:<20.4f} | {f_norm:<15.4f} | (Stable)")
    print(f"{'Gap Ratio':<15} | {b_gap:<20.1f}x             | {f_gap:<15.1f}x      | +{f_gap - b_gap:.1f}x")
    print(f"{'AUC':<15} | {b_auc:<20.4f} | {f_auc:<15.4f} | +{(f_auc-b_auc)*100:.2f}%")
    print("="*50)

    if f_auc > b_auc:
        print("✅ SUCCESS: Frozen backbone prevented catastrophic forgetting!")
    else:
        print("⚠️ NOTE: If AUC still drops, consider Prototype Metric Learning (Strategy 1).")

if __name__ == "__main__":
    main()