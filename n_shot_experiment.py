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

# === 🎛️ 核心参数控制台 ===
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
CLUSTER_FILE = "fault_clusters_analysis/fault_clustering_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 在这里调整你的策略！
SHOTS_PER_CLASS = 1   # <--- 修改这里！试试 1, 3, 5, 10
RANDOM_SEED = 42      # 固定种子，保证结果可复现 (论文里很重要)

# 实验分组
TRAIN_CLUSTERS = [0, 1, 2, 4]  # 从已知类里取 N-Shot
TEST_CLUSTERS = [3]            # 未知类 (全量测试)

# ==========================================
# 工具函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FaultFineTuneDataset(Dataset):
    """
    N-Shot 平衡策略：
    把珍贵的 N*4 个故障样本，复制几百倍，强行对齐正常样本的数量
    """
    def __init__(self, normal_windows, fault_windows):
        self.normal = normal_windows
        self.fault = fault_windows
        
        n_normal = len(self.normal)
        n_fault = len(self.fault)
        
        print(f"   ⚖️ {SHOTS_PER_CLASS}-Shot Balancing: Normal={n_normal}, Fault={n_fault}")
        
        if n_normal > n_fault and n_fault > 0:
            repeat_factor = int(n_normal / n_fault) + 1
            self.fault = np.tile(self.fault, (repeat_factor, 1, 1))[:n_normal]
            print(f"      -> 🔄 Replicated {n_fault} samples {repeat_factor}x to match normal data.")
            
        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([np.zeros(len(self.normal)), np.ones(len(self.fault))], axis=0)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

def get_n_shot_data(loader, cluster_df, target_clusters, shots=1):
    """
    从指定 Cluster 中，随机抽取 N 个不重叠的样本
    """
    collected_data = []
    print(f"🎲 Sampling {shots} shot(s) from Clusters {target_clusters}...")
    
    # 1. 构建候选池
    cluster_indices_map = {}
    for _, row in cluster_df.iterrows():
        cid = row['Cluster_Type']
        if cid not in target_clusters: continue
        if cid not in cluster_indices_map: cluster_indices_map[cid] = []
        
        # 取中间段，防止边缘噪声
        mid_point = (row['Start_Idx'] + row['End_Idx']) // 2
        cluster_indices_map[cid].append(mid_point)
        
    # 2. 随机抽样
    selected_indices = []
    for cid in target_clusters:
        if cid not in cluster_indices_map: continue
        candidates = cluster_indices_map[cid]
        
        # 如果样本不够，就全取
        k = min(len(candidates), shots)
        picks = random.sample(candidates, k)
        selected_indices.extend(picks)
        print(f"   🎯 Cluster {cid}: Picked {len(picks)} samples")
        
    # 3. 从 Loader 提取 (这是最耗时的步骤，但必须精准)
    selected_indices_set = set(selected_indices)
    current_idx = 0
    found_count = 0
    
    # 优化提取逻辑：只遍历一次
    for x in loader:
        batch_size = x.shape[0]
        batch_indices = range(current_idx, current_idx + batch_size)
        
        # 检查是否有交集
        common = selected_indices_set.intersection(batch_indices)
        if common:
            for pick in common:
                local_idx = pick - current_idx
                collected_data.append(x[local_idx].numpy())
                found_count += 1
                
        current_idx += batch_size
        if found_count >= len(selected_indices):
            break 
            
    return np.array(collected_data)

def load_full_cluster_data(loader, cluster_df, target_clusters):
    """加载全量数据用于评估"""
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

# ==========================================
# 主程序
# ==========================================
def main():
    set_seed(RANDOM_SEED)
    print(f"🚀 {SHOTS_PER_CLASS}-SHOT LEARNING EXPERIMENT START! (Seed={RANDOM_SEED})")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    # A. 正常数据 (Baseline)
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 10000: break 
        normal_data_pool.append(x.numpy())
    normal_data = np.concatenate(normal_data_pool)
    
    # B. 训练数据：N-Shot
    train_n_shot_data = get_n_shot_data(test_loader, df_clusters, TRAIN_CLUSTERS, shots=SHOTS_PER_CLASS)
    
    # C. 测试数据：全量未知故障
    test_novel_data = load_full_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    if len(train_n_shot_data) == 0:
        print("❌ Error: No training data found.")
        return

    # 2. 加载模型
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # 3. 极速微调
    print(f"\n⚡ Fine-tuning with {len(train_n_shot_data)} samples...")
    dataset = FaultFineTuneDataset(normal_data, train_n_shot_data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    model.train()
    for epoch in range(5): # 5个Epoch足够了
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ret = model(x)
            loss = deviation_loss(ret[0].squeeze(), ret[1][:,-1,:], x[:,-1,:], y, margin=5.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"   Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    # 4. 验证
    print("\n🧐 Evaluating on UNSEEN Novel Faults...")
    model.eval()
    
    # 计算 Novel Fault 分数
    scores_novel = []
    t_novel = torch.from_numpy(test_novel_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(test_novel_data), 256):
            b = t_novel[i:i+256]
            ret = model(b)
            l = torch.mean((ret[0].squeeze()-b[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-b[:,-1,:])**2, 1)
            scores_novel.append(l.cpu().numpy())
    scores_novel = np.concatenate(scores_novel)
    
    # 计算 Normal 分数
    scores_norm = []
    t_norm = torch.from_numpy(normal_data[:len(test_novel_data)]).float().to(DEVICE) # 1:1 对比
    with torch.no_grad():
        ret = model(t_norm)
        l = torch.mean((ret[0].squeeze()-t_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-t_norm[:,-1,:])**2, 1)
        scores_norm.append(l.cpu().numpy())
    scores_norm = np.concatenate(scores_norm)
    
    # 统计指标
    mean_novel = np.mean(scores_novel)
    mean_norm = np.mean(scores_norm)
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_novel))]),
        np.concatenate([scores_norm, scores_novel])
    )
    
    print(f"\n🏆 RESULT ({SHOTS_PER_CLASS}-Shot):")
    print(f"   Normal Score: {mean_norm:.4f}")
    print(f"   Novel Fault Score: {mean_novel:.4f}")
    print(f"   Gap Ratio: {mean_novel/mean_norm:.1f}x")
    print(f"   AUC: {auc:.4f}")
    
    # 判定
    if auc > 0.90:
        print("   ✅ Excellent! This is a publishable result.")
    else:
        print("   ⚠️ A bit low. Try increasing SHOTS_PER_CLASS.")

if __name__ == "__main__":
    main()