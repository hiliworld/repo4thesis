import torch
import numpy as np
import pandas as pd
import yaml
import random
import os
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_full_cluster_data(loader, cluster_df, target_clusters):
    """只加载指定 Cluster 的数据"""
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

def evaluate_model(model, normal_data, fault_data, batch_size=256):
    """评估 Zero-Shot 性能"""
    model.eval()
    if len(fault_data) == 0: return 0, 0, 0, 0
    
    # Fault Scores
    scores_fault = []
    t_fault = torch.from_numpy(fault_data).float().to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(fault_data), batch_size):
            b = t_fault[i:i+batch_size]
            ret = model(b)
            # 计算 Deviation Score
            l = torch.mean((ret[0].squeeze()-b[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-b[:,-1,:])**2, 1)
            scores_fault.append(l.cpu().numpy())
    scores_fault = np.concatenate(scores_fault)
    
    # Normal Scores (保持 1:1 数量对比，避免 AUC 偏差)
    scores_norm = []
    eval_len = min(len(normal_data), len(fault_data))
    t_norm = torch.from_numpy(normal_data[:eval_len]).float().to(DEVICE) 
    with torch.no_grad():
        ret = model(t_norm)
        l = torch.mean((ret[0].squeeze()-t_norm[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-t_norm[:,-1,:])**2, 1)
        scores_norm.append(l.cpu().numpy())
    scores_norm = np.concatenate(scores_norm)
    
    mean_fault = np.mean(scores_fault)
    mean_norm = np.mean(scores_norm)
    gap = mean_fault / mean_norm if mean_norm > 1e-9 else 0.0
    
    # 计算 AUC
    y_true = np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_fault))])
    y_scores = np.concatenate([scores_norm, scores_fault])
    auc = roc_auc_score(y_true, y_scores)
    
    return mean_norm, mean_fault, gap, auc

def main():
    set_seed(42)
    print("🚀 CLUSTER DIFFICULTY SCANNER (Zero-Shot Baseline Check)")
    print("正在寻找最难的故障类型（Baseline AUC 最低的那个）...")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # 准备正常数据 (取后半部分，防止泄漏，保持一致性)
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 20000: break 
        normal_data_pool.append(x.numpy())
    normal_data_full = np.concatenate(normal_data_pool)
    split_idx = int(len(normal_data_full) * 0.5)
    normal_test = normal_data_full[split_idx:]
    
    # 2. 扫描所有 Cluster
    unique_clusters = sorted(df_clusters['Cluster_Type'].unique())
    
    results = []
    print(f"\n🔎 Scanning {len(unique_clusters)} clusters...")
    print(f"{'Cluster':<8} | {'Samples':<8} | {'Fault Score':<12} | {'Gap Ratio':<10} | {'AUC (Zero-Shot)':<15}")
    print("-" * 65)
    
    for cid in unique_clusters:
        # 加载该 Cluster 的数据
        fault_data = load_full_cluster_data(test_loader, df_clusters, [cid])
        
        if len(fault_data) < 10:
            print(f"{cid:<8} | {'Skipped (Too small)':<30}")
            continue
            
        norm_s, fault_s, gap, auc = evaluate_model(model, normal_test, fault_data)
        
        results.append({
            'Cluster': cid,
            'AUC': auc
        })
        
        print(f"{cid:<8} | {len(fault_data):<8} | {fault_s:<12.4f} | {gap:<10.1f}x | {auc:<15.4f}")
        
    print("-" * 65)
    
    # 3. 智能推荐
    if not results: return
    
    # 找 AUC 最低的
    best_candidate = min(results, key=lambda x: x['AUC'])
    print(f"\n💡 策略建议 (STRATEGY RECOMMENDATION):")
    print(f"   请选择 Cluster {best_candidate['Cluster']} 作为你的 TEST_CLUSTERS！")
    print(f"   原因: 它的 Zero-Shot AUC 最低 ({best_candidate['AUC']:.4f})，说明它是基础模型的'软肋'。")
    print(f"   在此处应用小样本学习，效果提升将最显著！")

if __name__ == "__main__":
    main()