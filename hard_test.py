import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score

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
OUTPUT_DIR = "robustness_check_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 复用之前的工具函数
def load_cluster_data_indices(cluster_df, target_clusters):
    """只返回 indices，不加载数据，为了省内存"""
    target_indices = set()
    for _, row in cluster_df.iterrows():
        if row['Cluster_Type'] in target_clusters:
            target_indices.update(range(row['Start_Idx'], row['End_Idx']))
    return target_indices

class FaultFineTuneDataset(Dataset):
    """双向平衡数据集 (同 step15)"""
    def __init__(self, normal_windows, fault_windows):
        self.normal = normal_windows
        self.fault = fault_windows
        n_normal = len(self.normal)
        n_fault = len(self.fault)
        
        if n_fault > n_normal:
            repeat_factor = int(n_fault / n_normal) + 1
            self.normal = np.tile(self.normal, (repeat_factor, 1, 1))[:n_fault]
        elif n_normal > n_fault:
            repeat_factor = int(n_normal / n_fault) + 1
            self.fault = np.tile(self.fault, (repeat_factor, 1, 1))[:n_normal]
            
        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([np.zeros(len(self.normal)), np.ones(len(self.fault))], axis=0)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

def deviation_loss(pred, recon, target, labels, margin=5.0):
    error = torch.mean((pred - target)**2, dim=1) + torch.mean((recon - target)**2, dim=1)
    loss_normal = error[labels == 0].mean() if (labels==0).sum() > 0 else 0.0
    if (labels == 1).sum() > 0:
        loss_fault = torch.relu(margin - error[labels == 1]).mean()
    else:
        loss_fault = 0.0
    return loss_normal + loss_fault

# ==========================================
# 🔍 功能 1: 故障指纹相似度分析
# ==========================================
def analyze_cluster_similarity(loader, df_clusters, model):
    print("\n🔍 Analyzing Fault Heterogeneity (Are clusters actually different?)...")
    
    # 1. 计算每个 Cluster 的"平均故障向量" (Mean Fault Vector)
    # 我们用模型输出的 Latent Vector (Z) 代表故障的高维特征
    cluster_ids = sorted(df_clusters['Cluster_Type'].unique())
    cluster_vectors = {}
    
    model.eval()
    with torch.no_grad():
        # 遍历所有数据一次，收集特征
        current_idx = 0
        z_storage = []
        for x in loader:
            x = x.to(DEVICE)
            ret = model(x)
            z = ret[-1] # 取 latent vector [B, N, H] or [B, H]
            z_flat = z.view(z.shape[0], -1).cpu().numpy()
            z_storage.append(z_flat)
            
        all_z = np.concatenate(z_storage, axis=0)
    
    # 聚合
    for cid in cluster_ids:
        indices = load_cluster_data_indices(df_clusters, [cid])
        # 过滤掉越界的索引
        valid_indices = [i for i in indices if i < len(all_z)]
        if not valid_indices: continue
        
        # 计算该 Cluster 的中心向量
        centroid = np.mean(all_z[valid_indices], axis=0)
        cluster_vectors[cid] = centroid
        
    # 2. 计算余弦相似度矩阵
    matrix = np.zeros((len(cluster_vectors), len(cluster_vectors)))
    ids = list(cluster_vectors.keys())
    
    for i in range(len(ids)):
        for j in range(len(ids)):
            vec_i = cluster_vectors[ids[i]].reshape(1, -1)
            vec_j = cluster_vectors[ids[j]].reshape(1, -1)
            sim = cosine_similarity(vec_i, vec_j)[0][0]
            matrix[i, j] = sim
            
    # 3. 画热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, xticklabels=ids, yticklabels=ids, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Cluster Similarity Matrix (Latent Space)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.savefig(f"{OUTPUT_DIR}/cluster_similarity.png")
    print(f"   📸 Similarity heatmap saved to {OUTPUT_DIR}/cluster_similarity.png")
    
    # 4. 重点分析 Cluster 3 (未知) 和其他 Cluster 的距离
    if 3 in ids:
        idx_3 = ids.index(3)
        sims = matrix[idx_3]
        avg_sim = (np.sum(sims) - 1) / (len(ids) - 1) # 去掉自己
        print(f"   ⚠️ Similarity Check for Cluster 3 (Novel):")
        print(f"      Average Similarity to others: {avg_sim:.4f}")
        if avg_sim > 0.9:
            print("      🚨 WARNING: Cluster 3 is very similar to others. Leakage risk high!")
        else:
            print("      ✅ PASS: Cluster 3 is distinct from known clusters.")
            
# ==========================================
# 🧪 功能 2: 极限划分实验 (60% vs 40%)
# ==========================================
def run_hard_split_experiment(model, loader, df_clusters, feature_dim, normal_data):
    print("\n⚔️ Running Hard Split Experiment (60% Train / 40% Test)...")
    
    # === 重新划分 ===
    # Cluster 0 是最大的 (122个)，只用它做训练
    # Cluster 1, 2, 3, 4 全部当作未知！
    # 这样训练集占比约 122/169 ≈ 72% (或者更极端的 60%)
    
    # 方案：只训练 Cluster 0 和 2 (最大的两个)，测试 3
    # 或者更狠一点：只训练 Cluster 0，测试 Cluster 2 和 3
    
    TRAIN_C = [0]        # 仅使用最常见的故障训练
    TEST_C = [2, 3]      # 测试 次常见(2) 和 稀有(3)
    
    print(f"   🎓 Train Clusters: {TRAIN_C} (Base)")
    print(f"   📝 Test Clusters : {TEST_C}  (Novel)")
    
    # 提取数据
    train_data = []
    test_data = []
    
    # 这里为了代码简洁，直接复用之前的提取逻辑，但需要 loader 支持
    # 既然我们已经有 normal_data，还需要提取 fault data
    # 重新遍历一次 loader (为了稳健)
    current_idx = 0
    train_indices = load_cluster_data_indices(df_clusters, TRAIN_C)
    test_indices = load_cluster_data_indices(df_clusters, TEST_C)
    
    for x in loader:
        batch_size = x.shape[0]
        batch_indices = range(current_idx, current_idx + batch_size)
        
        # 提取 Train Data
        if not train_indices.isdisjoint(batch_indices):
            for i, g_idx in enumerate(batch_indices):
                if g_idx in train_indices: train_data.append(x[i].numpy())
                
        # 提取 Test Data
        if not test_indices.isdisjoint(batch_indices):
            for i, g_idx in enumerate(batch_indices):
                if g_idx in test_indices: test_data.append(x[i].numpy())
                
        current_idx += batch_size
        
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    print(f"   📦 Data Size -> Train Faults: {len(train_data)} | Test Faults: {len(test_data)}")
    
    # 微调
    dataset = FaultFineTuneDataset(normal_data, train_data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    model.train()
    for epoch in range(3): # 快速微调3轮
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ret = model(x)
            pred, recon = ret[0], ret[1]
            loss = deviation_loss(pred, recon[:, -1, :], x[:, -1, :], y, margin=5.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"      Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    # 验证
    model.eval()
    # 计算 Test Clusters 的分数
    scores = []
    tensor_test = torch.from_numpy(test_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(test_data), 256):
            batch = tensor_test[i:i+256]
            ret = model(batch)
            l = torch.mean((ret[0].squeeze()-batch[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-batch[:,-1,:])**2, 1)
            scores.append(l.cpu().numpy())
    
    fault_scores = np.concatenate(scores)
    avg_fault_score = np.mean(fault_scores)
    
    # 计算 AUC (Normal vs Novel Faults)
    # 我们用之前的 normal_data 的一部分作为负样本
    tensor_norm = torch.from_numpy(normal_data[:len(test_data)]).float().to(DEVICE) # 1:1 验证
    norm_scores_list = []
    with torch.no_grad():
         ret = model(tensor_norm)
         l = torch.mean((ret[0].squeeze()-tensor_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-tensor_norm[:,-1,:])**2, 1)
         norm_scores_list.append(l.cpu().numpy())
    norm_scores = np.concatenate(norm_scores_list)
    
    labels = np.concatenate([np.zeros(len(norm_scores)), np.ones(len(fault_scores))])
    all_scores = np.concatenate([norm_scores, fault_scores])
    auc = roc_auc_score(labels, all_scores)
    
    print(f"\n   🏆 Hard Split Result (Train {TRAIN_C} -> Test {TEST_C}):")
    print(f"      Normal Score: {np.mean(norm_scores):.4f}")
    print(f"      Novel Fault Score: {avg_fault_score:.4f}")
    print(f"      AUC: {auc:.4f}")
    
    if auc > 0.90:
        print("      ✅ Robustness Confirmed: Model generalizes well even with limited base classes.")
    else:
        print("      ⚠️ Robustness Warning: Model struggled with this split.")


def main():
    print("🚀 Starting Robustness Check & Heterogeneity Analysis...")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # 2. 准备 Normal Data 用于平衡
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 10000: break 
        normal_data_pool.append(x.numpy())
    normal_data = np.concatenate(normal_data_pool)
    
    # === 任务 1: 相似度分析 ===
    analyze_cluster_similarity(test_loader, df_clusters, model)
    
    # === 任务 2: 极限划分实验 ===
    # 重新加载纯净模型 (防止被之前的微调污染)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    run_hard_split_experiment(model, test_loader, df_clusters, feature_dim, normal_data)

if __name__ == "__main__":
    main()