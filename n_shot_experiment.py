import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import random
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# 尝试导入项目模块
try:
    from src.data.loader import get_dataloaders
    from src.models.anomaly_model import MyFinalModel
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    exit()

# ==========================================
# 🎛️ 实验配置
# ==========================================
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
CLUSTER_FILE = "fault_clusters_analysis/fault_ground_truth.csv" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🎯 实验目标：测试 Cluster 5 (新维度单点故障)
TRAIN_CLUSTERS = [0, 1, 2, 3, 4] 
TEST_CLUSTERS = [9]             

SHOTS_PER_CLASS = 1   # 1-Shot
RANDOM_SEED = 42

# ==========================================
# 🛠️ 工具函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        if cid in cluster_indices_map:
            candidates = cluster_indices_map[cid]
            if not candidates: continue
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

def get_features(model, data_loader):
    """
    关键函数：利用预训练模型提取特征 (Embedding)
    不经过最后的分类头 (Head)，直接取 encoder/gat 的输出
    """
    model.eval()
    features = []
    with torch.no_grad():
        for x in data_loader:
            x = x.to(DEVICE)
            # 1. 通过 Metric Encoder (Transformer/Conv)
            z = model.metric_encoder(x) 
            
            # 2. 如果有 GAT 层，也通过一下
            if hasattr(model, 'gat_layer'):
                z_gat, _ = model.gat_layer(z)
                z = z + z_gat # 残差连接
            
            # 3. Global Average Pooling: [Batch, Window, Feat] -> [Batch, Feat]
            # 我们需要一个固定长度的向量来代表这个样本
            z_flat = torch.mean(z, dim=1) 
            features.append(z_flat.cpu().numpy())
            
    return np.concatenate(features)

def evaluate_baseline(model, normal_data, fault_data):
    """Baseline: 使用原始的 Deviation Loss 或重构误差"""
    model.eval()
    loader_norm = DataLoader(normal_data, batch_size=256)
    loader_fault = DataLoader(fault_data, batch_size=256)
    
    def get_scores(loader):
        scores = []
        with torch.no_grad():
            for x in loader:
                x = x.to(DEVICE)
                ret = model(x)
                # 原始异常分计算方式 (Recon Loss + Pred Loss)
                l = torch.mean((ret[0].squeeze()-x[:,-1,:])**2, 1) + torch.mean((ret[1][:,-1,:]-x[:,-1,:])**2, 1)
                scores.append(l.cpu().numpy())
        return np.concatenate(scores)

    s_norm = get_scores(loader_norm)
    s_fault = get_scores(loader_fault)
    
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(s_norm)), np.ones(len(s_fault))]),
        np.concatenate([s_norm, s_fault])
    )
    return np.mean(s_norm), np.mean(s_fault), auc

# ==========================================
# 🚀 主程序
# ==========================================
def main():
    set_seed(RANDOM_SEED)
    print(f"🚀 1-SHOT PROTOTYPE EXPERIMENT (Strategy: Metric Learning)")
    print(f"   Target Cluster: {TEST_CLUSTERS}")
    
    # 1. 加载数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    if not os.path.exists(CLUSTER_FILE):
        print(f"❌ 没找到 {CLUSTER_FILE}")
        return
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    # 准备正常数据 (Split Half)
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 10000: break 
        normal_data_pool.append(x.numpy())
    normal_data_full = np.concatenate(normal_data_pool)
    split_idx = int(len(normal_data_full) * 0.5)
    normal_train = normal_data_full[:split_idx] # 用于计算 Normal Prototype
    normal_test  = normal_data_full[split_idx:] # 用于评估

    # 准备故障数据
    train_n_shot_data = get_n_shot_data(test_loader, df_clusters, TRAIN_CLUSTERS, shots=SHOTS_PER_CLASS)
    test_novel_data = load_full_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    if len(train_n_shot_data) == 0:
        print("❌ Error: No support data found.")
        return

    # 2. 加载模型
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # === STEP 0: Baseline Evaluation ===
    # 先看看如果不做任何处理，直接用原始模型跑分是多少
    print("\n📊 Baseline (Zero-Shot) Evaluation...")
    # 为了公平，Normal Test 取一部分
    eval_len = min(len(normal_test), len(test_novel_data))
    b_norm_data = normal_test[:eval_len]
    
    b_norm_score, b_fault_score, b_auc = evaluate_baseline(model, b_norm_data, test_novel_data)
    print(f"   [Base] AUC: {b_auc:.4f} | Fault Score: {b_fault_score:.4f} | Normal Score: {b_norm_score:.4f}")

    # === STEP 1: Calculate Prototypes (The "Learning" Phase) ===
    print("\n📐 Calculating Prototypes (No Gradient)...")
    
    # A. 计算 Normal Prototype (正常中心)
    # 使用所有 normal_train 数据，越丰富越准
    norm_loader = DataLoader(normal_train, batch_size=256)
    z_norm_all = get_features(model, norm_loader)
    proto_normal = np.mean(z_norm_all, axis=0)
    print(f"   Normal Prototype Shape: {proto_normal.shape}")
    
    # B. 计算 Fault Prototype (故障中心)
    # 使用那珍贵的 1-Shot 样本
    fault_loader = DataLoader(train_n_shot_data, batch_size=len(train_n_shot_data))
    z_fault_all = get_features(model, fault_loader)
    proto_fault = np.mean(z_fault_all, axis=0)
    print(f"   Fault Prototype Shape: {proto_fault.shape}")
    
    # === STEP 2: Prototypical Inference (The "Testing" Phase) ===
    print("\n🧐 Evaluating using Distance Metric...")
    
    def get_proto_score(data_array):
        loader = DataLoader(data_array, batch_size=256)
        z = get_features(model, loader)
        
        # 计算欧氏距离
        d_n = np.linalg.norm(z - proto_normal, axis=1) # 到正常的距离
        d_f = np.linalg.norm(z - proto_fault, axis=1)  # 到故障的距离
        
        # 核心公式：异常分 = d_n - d_f
        # 逻辑：如果离正常越远(d_n大)，离故障越近(d_f小)，那么 (d_n - d_f) 就越大 -> 越异常
        return d_n - d_f 
    
    # 对测试集进行打分
    scores_norm = get_proto_score(b_norm_data)
    scores_fault = get_proto_score(test_novel_data)
    
    # === 最终报告 ===
    f_norm_mean = np.mean(scores_norm)
    f_fault_mean = np.mean(scores_fault)
    
    f_auc = roc_auc_score(
        np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_fault))]),
        np.concatenate([scores_norm, scores_fault])
    )
    
    print("\n" + "="*50)
    print(f"🏆 PROTOTYPE STRATEGY RESULT (Cluster {TEST_CLUSTERS})")
    print("="*50)
    print(f"{'Metric':<15} | {'Baseline':<15} | {'Prototype':<15} | {'Change'}")
    print("-" * 65)
    # 注意：这里的 Score 是距离差，不是 absolute error，所以直接比较数值大小没意义，要看相对区分度
    print(f"{'Avg Fault':<15} | {b_fault_score:<15.4f} | {f_fault_mean:<15.4f} | (Distance Diff)")
    print(f"{'Avg Normal':<15} | {b_norm_score:<15.4f} | {f_norm_mean:<15.4f} | (Distance Diff)")
    print(f"{'AUC':<15} | {b_auc:<15.4f} | {f_auc:<15.4f} | +{(f_auc-b_auc)*100:.2f}%")
    print("="*50)
    
    if f_auc > b_auc:
        print("✅ SUCCESS: Prototype Metric is better than Zero-shot Baseline!")
    else:
        print("⚠️ NOTE: If AUC is lower, it means the feature space is not well-clustered.")

if __name__ == "__main__":
    main()