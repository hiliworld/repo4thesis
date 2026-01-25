import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score

# === 导入 src 模块 ===
try:
    from src.data.loader import get_dataloaders
    from src.models.anomaly_model import MyFinalModel
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    exit()

# === 配置 ===
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "diagnosis_results_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 路径硬编码
HARDCODED_LABEL_PATH = "/home/sde/MyThesis/data/ServerMachineDataset/test_label"

def load_labels(config):
    """加载标签"""
    test_path_conf = config['dataset']['test_file']
    if "test" in test_path_conf:
        label_path = test_path_conf.replace("/test/", "/test_label/").replace("/test", "/test_label")
    else:
        label_path = HARDCODED_LABEL_PATH
    
    if not os.path.exists(label_path): label_path = HARDCODED_LABEL_PATH
    pattern = config['dataset']['format']['pattern']
    files = sorted(glob.glob(os.path.join(label_path, pattern)))
    label_list = []
    window = config['dataset']['window_size']
    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            raw = df.values.flatten()
            if len(raw) > window: label_list.append(raw[window:])
        except: pass
    if not label_list: return None
    return np.concatenate(label_list)

def get_best_threshold(scores, labels):
    """计算最佳 F1 阈值"""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def plot_final_diagnosis(mat_orig, mat_recon, mat_pred, 
                         scores, labels, threshold, 
                         start_idx, length=500, title="Diagnosis"):
    """
    三图流绘制：Recon, Pred, Score+Threshold
    """
    end_idx = min(start_idx + length, len(scores))
    if start_idx >= end_idx: return
    time_steps = np.arange(start_idx, end_idx)
    
    # 切片
    s_orig  = mat_orig[start_idx:end_idx]
    s_recon = mat_recon[start_idx:end_idx]
    s_pred  = mat_pred[start_idx:end_idx]
    s_score = scores[start_idx:end_idx]
    s_label = labels[start_idx:end_idx]
    
    # === 智能特征选择 ===
    # 找出这段时间内误差最大的特征
    err_matrix = (s_pred - s_orig)**2 + (s_recon - s_orig)**2
    feat_errors = np.mean(err_matrix, axis=0)
    top_feat_idx = np.argmax(feat_errors)
    
    # 提取该特征的波形
    f_orig = s_orig[:, top_feat_idx]
    f_recon = s_recon[:, top_feat_idx]
    f_pred = s_pred[:, top_feat_idx]
    
    # === 绘图 ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Row 1: Reconstruction Check
    axes[0].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original (Feat {top_feat_idx})')
    axes[0].plot(time_steps, f_recon, color='green', linestyle='--', linewidth=1.5, label='Reconstruction')
    axes[0].set_title(f"Reconstruction Fit (Feature {top_feat_idx})")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Row 2: Prediction Check
    axes[0].sharex(axes[1])
    axes[1].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original (Feat {top_feat_idx})')
    axes[1].plot(time_steps, f_pred, color='orange', linestyle=':', linewidth=2, label='Prediction')
    axes[1].set_title(f"Prediction Fit (Feature {top_feat_idx})")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Row 3: Decision View (Score vs Threshold)
    # 不做归一化，直接画原始分数，这样阈值才有意义
    axes[2].plot(time_steps, s_score, color='blue', linewidth=1.5, label='Anomaly Score')
    
    # 画阈值线
    axes[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    # 画 Ground Truth 阴影
    axes[2].fill_between(time_steps, 0, s_score.max(), where=(s_label > 0.5), 
                         color='red', alpha=0.2, label='Ground Truth')
    
    axes[2].set_title("Anomaly Score & Decision Threshold")
    axes[2].legend(loc='upper right')
    axes[2].set_xlabel("Time Steps")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{title}_idx{start_idx}.png"
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"🔥 Final Visualizer (3-Panel + Threshold) | Device: {DEVICE}")
    
    # 1. 准备数据
    try:
        _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 准备模型
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    
    model = MyFinalModel(config).to(DEVICE)
    if not os.path.exists(MODEL_NAME):
        print(f"❌ 未找到模型: {MODEL_NAME}")
        return
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    model.eval()
    
    # 3. 全量推理
    print("🚀 Running Inference to calculate Global Threshold...")
    orig_list, recon_list, pred_list = [], [], []
    score_list = []
    
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Inference"):
            x = x.to(DEVICE)
            pred_next, recon_window, _ = model(x)
            
            # Dimensions
            target_curr = x[:, -1, :]
            recon_curr  = recon_window[:, -1, :]
            pred_curr   = pred_next
            if pred_curr.dim() == 3: pred_curr = pred_curr.squeeze(-1)
            
            # Score
            l_pred = torch.mean((pred_curr - target_curr) ** 2, dim=1)
            l_recon = torch.mean((recon_curr - target_curr) ** 2, dim=1)
            score = l_pred + l_recon
            
            # Store
            orig_list.append(target_curr.cpu().numpy())
            recon_list.append(recon_curr.cpu().numpy())
            pred_list.append(pred_curr.cpu().numpy())
            score_list.append(score.cpu().numpy())
            
    # Concat
    mat_orig  = np.concatenate(orig_list, axis=0)
    mat_recon = np.concatenate(recon_list, axis=0)
    mat_pred  = np.concatenate(pred_list, axis=0)
    scores    = np.concatenate(score_list, axis=0)
    
    # 4. 加载标签 & 对齐
    labels = load_labels(config)
    if labels is None: return
    
    min_len = min(len(scores), len(labels))
    mat_orig = mat_orig[:min_len]
    mat_recon = mat_recon[:min_len]
    mat_pred = mat_pred[:min_len]
    scores = scores[:min_len]
    labels = labels[:min_len]
    
    # 5. 计算最佳阈值 (关键步骤)
    print("⚖️  正在计算最佳 F1 阈值...")
    best_thresh, best_f1 = get_best_threshold(scores, labels)
    print(f"   🏆 Best Threshold: {best_thresh:.6f} (F1: {best_f1:.4f})")
    
    # 6. 绘图循环
    # A. 真实故障
    diff = np.diff(labels, prepend=0)
    starts = np.where(diff == 1)[0]
    print(f"🔎 绘制前 5 个真实故障...")
    for i, start in enumerate(starts[:5]):
        plot_final_diagnosis(mat_orig, mat_recon, mat_pred, scores, labels, 
                             threshold=best_thresh,
                             start_idx=max(0, start - 50), 
                             title=f"True_Anomaly_{i+1}")
                             
    # B. 误报 (Top False Positives)
    # 定义误报：Label=0 且 Score > Threshold
    # 我们找 Score 超过 Threshold 最多的几个点
    fp_mask = (labels == 0) & (scores > best_thresh)
    fp_indices = np.where(fp_mask)[0]
    
    if len(fp_indices) > 0:
        # 按分数排序，取最大的
        fp_scores = scores[fp_indices]
        sorted_fp_idx = fp_indices[np.argsort(fp_scores)[-3:]] # Top 3
        
        print(f"🔎 绘制 Top 3 误报 (超过阈值)...")
        for i, idx in enumerate(sorted_fp_idx):
             plot_final_diagnosis(mat_orig, mat_recon, mat_pred, scores, labels, 
                                  threshold=best_thresh,
                                  start_idx=max(0, idx - 250), 
                                  title=f"False_Positive_{i+1}")
    else:
        print("🎉 厉害！没有发现误报 (没有 Label=0 且 Score>Threshold 的点)")

    print(f"\n✅ 图片已保存至 {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()