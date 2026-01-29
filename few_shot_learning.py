import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import yaml
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

# === 实验分组 ===
TRAIN_CLUSTERS = [0, 1, 2, 4]  # 80% 已知
TEST_CLUSTERS = [3]            # 20% 未知

# ==========================================
# 1. 数据集定义
# ==========================================
class FaultFineTuneDataset(Dataset):
    """
    [修改] 双向数据平衡：谁少就复制谁，确保 1:1 比例
    """
    def __init__(self, normal_windows, fault_windows):
        self.normal = normal_windows
        self.fault = fault_windows
        
        n_normal = len(self.normal)
        n_fault = len(self.fault)
        
        print(f"   ⚖️ Balancing Data: Normal={n_normal}, Fault={n_fault}")
        
        # 策略：以数量多的一方为基准，复制数量少的一方
        if n_fault > n_normal:
            # 故障多，复制正常样本
            repeat_factor = int(n_fault / n_normal) + 1
            self.normal = np.tile(self.normal, (repeat_factor, 1, 1))[:n_fault] # 截断至相同数量
            print(f"      -> Upsampled Normal to {len(self.normal)}")
        elif n_normal > n_fault:
            # 正常多，复制故障样本
            repeat_factor = int(n_normal / n_fault) + 1
            self.fault = np.tile(self.fault, (repeat_factor, 1, 1))[:n_normal]
            print(f"      -> Upsampled Fault to {len(self.fault)}")
            
        # 现在两者数量一致了
        self.data = np.concatenate([self.normal, self.fault], axis=0)
        self.labels = np.concatenate([
            np.zeros(len(self.normal)), 
            np.ones(len(self.fault))
        ], axis=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()

# ==========================================
# 2. 工具函数
# ==========================================
def load_cluster_data(loader, cluster_df, target_clusters):
    """提取指定 Cluster 的数据"""
    target_indices = set()
    for _, row in cluster_df.iterrows():
        if row['Cluster_Type'] in target_clusters:
            # 扩大一点范围，确保能覆盖故障全貌
            target_indices.update(range(row['Start_Idx'], row['End_Idx']))
            
    collected_data = []
    current_idx = 0
    
    print(f"📥 Extracting data for Clusters {target_clusters}...")
    for x in loader:
        batch_size = x.shape[0]
        batch_indices = range(current_idx, current_idx + batch_size)
        
        # 检查交集
        if not target_indices.isdisjoint(batch_indices):
            for i, global_idx in enumerate(batch_indices):
                if global_idx in target_indices:
                    collected_data.append(x[i].numpy())
        current_idx += batch_size
        
    return np.array(collected_data)

def evaluate_on_clusters(model, data, title="Eval"):
    """
    [升级版] 评估模型，并拆解分数的来源
    """
    model.eval()
    if len(data) == 0: return 0.0
    
    tensor_data = torch.from_numpy(data).float().to(DEVICE)
    batch_size = 256
    
    total_pred_loss = []
    total_recon_loss = []
    total_combined_loss = []
    feature_wise_errors = [] # 用于存储每个特征的平均误差
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = tensor_data[i : i+batch_size]
            ret = model(batch)
            pred = ret[0]
            recon = ret[1]
            
            target = batch[:, -1, :]
            # 取最后一个时间步
            recon_last = recon[:, -1, :]
            pred_last = pred.squeeze(-1) if pred.dim() == 3 else pred
            
            # 1. 拆解计算各项损失 [Batch, Features]
            loss_pred_per_feat = (pred_last - target)**2
            loss_recon_per_feat = (recon_last - target)**2
            
            # 2. 记录总分
            # [Batch]
            l_pred = torch.mean(loss_pred_per_feat, dim=1)
            l_recon = torch.mean(loss_recon_per_feat, dim=1)
            l_comb = l_pred + l_recon
            
            total_pred_loss.append(l_pred.cpu().numpy())
            total_recon_loss.append(l_recon.cpu().numpy())
            total_combined_loss.append(l_comb.cpu().numpy())
            
            # 3. 记录特征级误差
            feat_err = (loss_pred_per_feat + loss_recon_per_feat).cpu().numpy()
            feature_wise_errors.append(np.mean(feat_err, axis=0)) 
            
    # === 统计环节 ===
    avg_pred = np.mean(np.concatenate(total_pred_loss))
    avg_recon = np.mean(np.concatenate(total_recon_loss))
    avg_total = np.mean(np.concatenate(total_combined_loss))
    
    # 计算所有 batch 的平均特征误差
    avg_feat_errors = np.mean(np.array(feature_wise_errors), axis=0)
    
    # 找出贡献最大的 Top-3 特征
    top3_idx = np.argsort(avg_feat_errors)[::-1][:3]
    top3_vals = avg_feat_errors[top3_idx]
    
    print(f"\n📊 [{title}] Analysis Breakdown:")
    print(f"   🔹 Total Score: {avg_total:.6f}")
    print(f"   🔹 Composition: Pred={avg_pred:.6f} | Recon={avg_recon:.6f}")
    print(f"   🔹 Top Culprits (Features contributing most error):")
    for idx, val in zip(top3_idx, top3_vals):
        print(f"      - Feat {idx}: Loss = {val:.4f}")

    return avg_total

def verify_performance(model, normal_data, fault_data, title="Verification"):
    """
    科学验证：计算 AUC 和 F1，并检查正常样本是否被误伤
    """
    model.eval()
    
    # 1. 准备混合数据
    n_normal = len(normal_data)
    n_fault = len(fault_data)
    
    data = np.concatenate([normal_data, fault_data], axis=0)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fault)], axis=0)
    
    tensor_data = torch.from_numpy(data).float().to(DEVICE)
    scores = []
    
    # 2. 推理计算分数
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = tensor_data[i : i+batch_size]
            ret = model(batch)
            pred, recon = ret[0], ret[1]
            
            target = batch[:, -1, :]
            recon_last = recon[:, -1, :] 
            pred_last = pred.squeeze(-1) if pred.dim() == 3 else pred
            
            loss = torch.mean((pred_last - target)**2, dim=1) + torch.mean((recon_last - target)**2, dim=1)
            scores.append(loss.cpu().numpy())
            
    all_scores = np.concatenate(scores)
    
    # 3. 分析
    scores_normal = all_scores[:n_normal]
    scores_fault = all_scores[n_normal:]
    
    mean_norm = np.mean(scores_normal)
    mean_fault = np.mean(scores_fault)
    
    # 4. 计算指标
    auc = roc_auc_score(labels, all_scores)
    
    prec, rec, thresholds = precision_recall_curve(labels, all_scores)
    f1_scores = 2 * rec * prec / (rec + prec + 1e-10)
    best_f1 = np.max(f1_scores)
    
    print(f"\n📊 [{title}] Rigorous Validation:")
    print(f"   🔹 Normal Score (Avg): {mean_norm:.6f}")
    print(f"   🔹 Fault  Score (Avg): {mean_fault:.6f}")
    print(f"   🔹 Gap Ratio (Fault/Normal): {mean_fault/mean_norm:.1f}x")
    print(f"   🏆 AUC Score: {auc:.6f} (1.0 is perfect)")
    print(f"   🏆 Best F1  : {best_f1:.6f}")
    
    print("\n   [Score Distribution Visualization]")
    print(f"   Normal: [{'#' * int(min(mean_norm*10, 20))}] ({mean_norm:.4f})")
    print(f"   Fault : [{'!' * int(min(mean_fault*10, 20))}] ({mean_fault:.4f})")
    
    return auc, mean_norm, mean_fault

def deviation_loss(pred, recon, target, labels, margin=5.0):
    """Deviation Loss"""
    error = torch.mean((pred - target)**2, dim=1) + torch.mean((recon - target)**2, dim=1)
    loss_normal = error[labels == 0].mean() if (labels==0).sum() > 0 else 0.0
    
    if (labels == 1).sum() > 0:
        loss_fault = torch.relu(margin - error[labels == 1]).mean()
    else:
        loss_fault = 0.0
        
    return loss_normal + loss_fault

# ==========================================
# 3. 主程序
# ==========================================
def main():
    print("🚀 Few-Shot Generalization Experiment Start!")
    
    # 1. 准备数据
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    print("📦 Harvesting Normal Data...")
    normal_data_pool = []
    for i, x in enumerate(test_loader):
        if i * x.shape[0] > 30000: break 
        normal_data_pool.append(x.numpy())
    normal_data = np.concatenate(normal_data_pool)
    
    train_fault_data = load_cluster_data(test_loader, df_clusters, TRAIN_CLUSTERS)
    test_fault_data = load_cluster_data(test_loader, df_clusters, TEST_CLUSTERS)
    
    print(f"\n📦 Data Summary:")
    print(f"   Normal Data Pool: {len(normal_data)}")
    print(f"   Known Faults (Train): {len(train_fault_data)} samples")
    print(f"   Unknown Faults (Test): {len(test_fault_data)} samples")
    
    # 2. 加载模型
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    
    # Step 1: Baseline
    print("\n🧐 Step 1: Evaluating Baseline...")
    score_before = evaluate_on_clusters(model, test_fault_data, title="Baseline")
    
    # Step 2: Fine-tuning
    print("\n🎓 Step 2: Supervised Fine-tuning...")
    dataset = FaultFineTuneDataset(normal_data, train_fault_data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5) 
    
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            ret = model(x)
            pred = ret[0]
            recon = ret[1]
            recon_last = recon[:, -1, :] 
            pred = pred.squeeze(-1) if pred.dim() == 3 else pred
            target = x[:, -1, :]
            
            loss = deviation_loss(pred, recon_last, target, y, margin=5.0)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    # Step 3: Evaluation
    print("\n😎 Step 3: Evaluating After Fine-tuning...")
    score_after = evaluate_on_clusters(model, test_fault_data, title="Fine-tuned")
    
    print("\n" + "="*40)
    print("🏆 EXPERIMENT RESULT 🏆")
    print("="*40)
    print(f"Unknown Fault Sensitivity (Cluster {TEST_CLUSTERS}):")
    print(f"   Before: {score_before:.6f}")
    print(f"   After : {score_after:.6f}")
    
    improvement = (score_after - score_before) / score_before * 100
    if score_after > score_before:
        print(f"\n✅ SUCCESS! Improved by {improvement:.2f}%")
        print("结论: Meta-Learning 有效！模型学会了'故障'的概念。")
    else:
        print(f"\n❌ No Improvement. ({improvement:.2f}%)")

    # ==========================================
    # 🔥 科学验证阶段
    # ==========================================
    print("\n" + "="*40)
    print("🧪 SCIENTIFIC VALIDATION PHASE")
    print("="*40)
    
    # 验证微调后的模型
    verify_performance(model, normal_data, test_fault_data, title="Post-Training Check")

    print("\n💡 解读指南：")
    print("1. 如果 AUC > 0.95 且 Normal Score < 0.1：完美！(模型学会了抓坏人，且没误伤好人)")
    print("2. 如果 Normal Score 也变成了 10.0：失败。(模型这叫'摆烂'，它把所有东西都判成了故障)")
    print("3. 如果 Gap Ratio > 100x：这在论文里是非常强有力的证据。")

if __name__ == "__main__":
    main()