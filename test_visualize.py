import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import glob
import matplotlib.pyplot as plt
import seaborn as sns
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
OUTPUT_DIR = "diagnosis_results_heatmap_fix" # 新目录
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

def select_top_k_events(indices, scores, k=10, min_dist=500, mode='max'):
    """独立事件筛选"""
    if len(indices) == 0: return []
    target_scores = scores[indices]
    if mode == 'max':
        sorted_idx_positions = np.argsort(target_scores)[::-1]
    else:
        sorted_idx_positions = np.argsort(target_scores)
    sorted_indices = indices[sorted_idx_positions]
    
    selected_indices = []
    for idx in sorted_indices:
        if len(selected_indices) >= k: break
        is_close = False
        for selected in selected_indices:
            if abs(idx - selected) < min_dist:
                is_close = True
                break
        if not is_close: selected_indices.append(idx)
    return selected_indices

def plot_diagnosis_with_heatmap(mat_orig, mat_recon, mat_pred, 
                                scores, labels, threshold, 
                                start_idx, length=500, title="Diagnosis"):
    """
    4 图流：Recon, Pred, Decision, Heatmap
    """
    end_idx = min(start_idx + length, len(scores))
    if start_idx >= end_idx: return
    
    # === 【关键修复】使用相对坐标 (0, 1, 2...) 而不是绝对坐标 ===
    # 这样才能和 Seaborn Heatmap 的坐标对齐
    plot_len = end_idx - start_idx
    time_steps = np.arange(plot_len)
    
    # 切片
    s_orig  = mat_orig[start_idx:end_idx]
    s_recon = mat_recon[start_idx:end_idx]
    s_pred  = mat_pred[start_idx:end_idx]
    s_score = scores[start_idx:end_idx]
    s_label = labels[start_idx:end_idx]
    
    # 计算全量特征误差矩阵 [Time, Features]
    err_matrix = (s_pred - s_orig)**2 + (s_recon - s_orig)**2
    
    # 找到最大误差特征
    feat_errors = np.mean(err_matrix, axis=0)
    top_feat_idx = np.argmax(feat_errors)
    
    f_orig = s_orig[:, top_feat_idx]
    f_recon = s_recon[:, top_feat_idx]
    f_pred = s_pred[:, top_feat_idx]
    
    # === 绘图 (4 行) ===
    # sharex=True: 现在大家都是 0-500，可以安全共享了
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Row 1: Reconstruction
    axes[0].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original (Feat {top_feat_idx})')
    axes[0].plot(time_steps, f_recon, color='green', linestyle='--', linewidth=1.5, label='Reconstruction')
    axes[0].set_title(f"{title} | Recon Fit (Feat {top_feat_idx}) | Start Idx: {start_idx}")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Row 2: Prediction
    axes[1].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original')
    axes[1].plot(time_steps, f_pred, color='orange', linestyle=':', linewidth=2, label='Prediction')
    axes[1].set_title(f"{title} | Pred Fit (Feat {top_feat_idx})")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Row 3: Decision
    axes[2].plot(time_steps, s_score, color='blue', linewidth=1.5, label='Anomaly Score')
    axes[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    axes[2].fill_between(time_steps, 0, s_score.max(), where=(s_label > 0.5), 
                         color='red', alpha=0.2, label='Ground Truth')
    axes[2].set_title("Anomaly Score & Decision")
    axes[2].grid(True, alpha=0.3)
    
    # Row 4: Heatmap (全景图)
    # 转置为 [Features, Time]
    heatmap_data = err_matrix.T 
    
    # 视觉优化
    vmax = np.percentile(heatmap_data, 99) if len(heatmap_data) > 0 else 1.0
    
    sns.heatmap(heatmap_data, ax=axes[3], cmap="Reds", cbar=False, vmin=0, vmax=vmax)
    axes[3].set_title(f"Global Error Heatmap (All {heatmap_data.shape[0]} Features)")
    axes[3].set_ylabel("Feature Index")
    axes[3].set_xlabel(f"Time Steps (+{start_idx})")
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{title}_idx{start_idx}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   📸 Saved: {title}")

def main():
    print(f"🔥 Diagnosis with Heatmap Fix | Device: {DEVICE}")
    
    # 1. 加载数据
    try:
        _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    
    # 2. 加载模型
    model = MyFinalModel(config).to(DEVICE)
    if not os.path.exists(MODEL_NAME):
        print(f"❌ 未找到模型: {MODEL_NAME}")
        return
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    model.eval()
    
    # 3. 全量推理
    print("🚀 Running Inference...")
    orig_list, recon_list, pred_list, score_list = [], [], [], []
    
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Inference"):
            x = x.to(DEVICE)
            pred_next, recon_window, _ = model(x)
            
            target_curr = x[:, -1, :]
            recon_curr  = recon_window[:, -1, :]
            pred_curr   = pred_next
            if pred_curr.dim() == 3: pred_curr = pred_curr.squeeze(-1)
            
            l_pred = torch.mean((pred_curr - target_curr) ** 2, dim=1)
            l_recon = torch.mean((recon_curr - target_curr) ** 2, dim=1)
            score = l_pred + l_recon
            
            orig_list.append(target_curr.cpu().numpy())
            recon_list.append(recon_curr.cpu().numpy())
            pred_list.append(pred_curr.cpu().numpy())
            score_list.append(score.cpu().numpy())
            
    mat_orig  = np.concatenate(orig_list, axis=0)
    mat_recon = np.concatenate(recon_list, axis=0)
    mat_pred  = np.concatenate(pred_list, axis=0)
    scores    = np.concatenate(score_list, axis=0)
    
    # 4. 标签对齐
    labels = load_labels(config)
    if labels is None: return
    
    min_len = min(len(scores), len(labels))
    mat_orig = mat_orig[:min_len]
    mat_recon = mat_recon[:min_len]
    mat_pred = mat_pred[:min_len]
    scores = scores[:min_len]
    labels = labels[:min_len]
    
    # 5. 计算阈值
    print("⚖️  Calculating Best Threshold...")
    best_thresh, best_f1 = get_best_threshold(scores, labels)
    print(f"   🏆 Threshold: {best_thresh:.6f} | Best F1: {best_f1:.4f}")
    
    # ==========================================
    # 🔍 核心逻辑：Top-K 错误分析
    # ==========================================
    
    # A. 分析误报 (FP)
    fp_indices = np.where((labels == 0) & (scores > best_thresh))[0]
    top_fps = select_top_k_events(fp_indices, scores, k=10, mode='max')
    
    print(f"\n🔎 绘制 Top 10 误报 (FP)...")
    for i, idx in enumerate(top_fps):
        plot_diagnosis_with_heatmap(mat_orig, mat_recon, mat_pred, scores, labels, 
                             threshold=best_thresh,
                             start_idx=max(0, idx - 250), 
                             title=f"FP_Rank{i+1}_Score{scores[idx]:.2f}")

    # B. 分析漏报 (FN)
    fn_indices = np.where((labels == 1) & (scores <= best_thresh))[0]
    top_fns = select_top_k_events(fn_indices, scores, k=10, mode='min')
    
    print(f"\n🔎 绘制 Top 10 漏报 (FN)...")
    for i, idx in enumerate(top_fns):
        plot_diagnosis_with_heatmap(mat_orig, mat_recon, mat_pred, scores, labels, 
                             threshold=best_thresh,
                             start_idx=max(0, idx - 250), 
                             title=f"FN_Rank{i+1}_Score{scores[idx]:.2f}")

    print(f"\n✅ 分析完成！请查看 {OUTPUT_DIR}/ 文件夹。")

if __name__ == "__main__":
    main()