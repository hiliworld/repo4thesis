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
OUTPUT_DIR = "diagnosis_results_integrated"  # 新的输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 路径硬编码
HARDCODED_LABEL_PATH = "/home/sde/MyThesis/data/ServerMachineDataset/test_label"

# ==========================================
# 🔧 工具函数区
# ==========================================

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

def robust_smoothing(scores, window_size=20):
    """
    Max Pooling + Moving Average 平滑策略
    """
    s = pd.Series(scores)
    # 1. Max Pooling: 填补漏报坑
    s_max = s.rolling(window=window_size, center=True, min_periods=1).max()
    # 2. Mean Smoothing: 边缘平滑
    s_final = s_max.rolling(window=int(window_size/2), center=True, min_periods=1).mean()
    return s_final.values

def select_top_k_events(indices, scores, k=10, min_dist=500, mode='max'):
    """独立事件筛选"""
    if len(indices) == 0: return []
    target_scores = scores[indices]
    if mode == 'max': # 找误报 (分数高的)
        sorted_idx_positions = np.argsort(target_scores)[::-1]
    else:             # 找漏报 (分数低的)
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

def analyze_root_cause_in_window(err_matrix):
    """
    [新功能] 微观诊断：分析这段窗口内的根因
    :return: 文本摘要 (Top-3 特征)
    """
    # 1. 计算每个特征的总误差贡献
    total_feat_error = np.sum(err_matrix, axis=0)
    top_3_idx = np.argsort(total_feat_error)[::-1][:3]
    
    # 2. 计算稳定性 (Top-1 特征出现的频次)
    # 在这个时间窗口内，每个时刻谁是 Top-1？
    top1_per_step = np.argmax(err_matrix, axis=1)
    # 统计出现最多的特征
    counts = np.bincount(top1_per_step, minlength=err_matrix.shape[1])
    dominant_feat = np.argmax(counts)
    dominance_ratio = counts[dominant_feat] / err_matrix.shape[0]
    
    # 生成报告字符串
    report = f"Top Culprits: Feat {top_3_idx[0]}, {top_3_idx[1]}, {top_3_idx[2]} | "
    if dominance_ratio > 0.8:
        report += f"Stable Root Cause: Feat {dominant_feat} ({dominance_ratio*100:.1f}%)"
    else:
        report += f"Unstable (Switching): Dominated by Feat {dominant_feat} ({dominance_ratio*100:.1f}%)"
        
    return report, top_3_idx[0]

# ==========================================
# 📊 绘图核心函数
# ==========================================

def plot_integrated_diagnosis(mat_orig, mat_recon, mat_pred, 
                              scores, labels, threshold, 
                              start_idx, length=500, title="Diagnosis"):
    """
    集成版绘图：Recon + Pred + Decision + Heatmap + Root Cause Text
    """
    end_idx = min(start_idx + length, len(scores))
    if start_idx >= end_idx: return
    
    # === 相对坐标 (0, 1, 2...) ===
    plot_len = end_idx - start_idx
    time_steps = np.arange(plot_len)
    
    # 切片
    s_orig  = mat_orig[start_idx:end_idx]
    s_recon = mat_recon[start_idx:end_idx]
    s_pred  = mat_pred[start_idx:end_idx]
    s_score = scores[start_idx:end_idx]
    s_label = labels[start_idx:end_idx]
    
    # 计算误差矩阵 [Time, Features]
    err_matrix = (s_pred - s_orig)**2 + (s_recon - s_orig)**2
    
    # === [新] 调用微观诊断 ===
    # 计算这段时间内最严重的特征，以及根因报告
    root_cause_text, top_feat_idx = analyze_root_cause_in_window(err_matrix)
    print(f"   🕵️‍♂️ {title}: {root_cause_text}")
    
    # 提取 Top-1 特征的波形用于绘制前两行
    f_orig = s_orig[:, top_feat_idx]
    f_recon = s_recon[:, top_feat_idx]
    f_pred = s_pred[:, top_feat_idx]
    
    # === 绘图布局 (4行) ===
    # 关键修改：前3个共享X轴，第4个(Heatmap)不共享，防止干扰
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    
    # Row 1: Reconstruction
    axes[0].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original (Feat {top_feat_idx})')
    axes[0].plot(time_steps, f_recon, color='green', linestyle='--', linewidth=1.5, label='Reconstruction')
    axes[0].set_title(f"{title} | Recon Fit (Worst Feat {top_feat_idx})")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    # 共享 X 轴逻辑手动处理
    axes[0].set_xlim(0, plot_len)

    # Row 2: Prediction
    axes[1].plot(time_steps, f_orig, color='black', alpha=0.6, label=f'Original')
    axes[1].plot(time_steps, f_pred, color='orange', linestyle=':', linewidth=2, label='Prediction')
    axes[1].set_title(f"Prediction Fit (Worst Feat {top_feat_idx})")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, plot_len)
    
    # Row 3: Decision (Score)
    axes[2].plot(time_steps, s_score, color='blue', linewidth=1.5, label='Anomaly Score')
    axes[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    axes[2].fill_between(time_steps, 0, s_score.max(), where=(s_label > 0.5), 
                         color='red', alpha=0.2, label='Ground Truth')
    axes[2].set_title(f"Anomaly Score | {root_cause_text}") # 把诊断结果写在标题里
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, plot_len)
    
    # Row 4: Heatmap
    # ⚠️ 独立坐标轴，防止 Seaborn 和 Matplotlib 打架
    heatmap_data = err_matrix.T 
    vmax = np.percentile(heatmap_data, 99) if len(heatmap_data) > 0 else 1.0
    
    sns.heatmap(heatmap_data, ax=axes[3], cmap="Reds", cbar=False, vmin=0, vmax=vmax, 
                xticklabels=50) # 每50个点显示一个刻度
    axes[3].set_title(f"Global Error Heatmap (All {heatmap_data.shape[0]} Features)")
    axes[3].set_ylabel("Feature Index")
    axes[3].set_xlabel(f"Time Steps (+{start_idx})")
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{title}_idx{start_idx}.png"
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 🚀 主程序
# ==========================================

def main():
    print(f"🔥 Integrated Diagnosis (Smooth + RootCause) | Device: {DEVICE}")
    
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
            # 适配你的模型返回值：可能是 3 个 (Pred, Recon, Latent)
            ret = model(x)
            pred_next = ret[0]
            recon_window = ret[1]
            
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
    
    # ==============================
    # 🔥 鲁棒平滑 (Max Pooling)
    # ==============================
    SMOOTH_WINDOW = 50 
    print(f"🧹 Applying Max-Pooling Smoothing (Window={SMOOTH_WINDOW})...")
    scores = robust_smoothing(scores, window_size=SMOOTH_WINDOW)
    
    # 5. 计算最佳阈值
    print("⚖️  Calculating Best Threshold (on smoothed scores)...")
    best_thresh, best_f1 = get_best_threshold(scores, labels)
    print(f"   🏆 Threshold: {best_thresh:.6f} | Best F1: {best_f1:.4f}")
    
    # ==========================================
    # 🔍 诊断循环
    # ==========================================
    
    # A. 分析误报 (FP)
    fp_indices = np.where((labels == 0) & (scores > best_thresh))[0]
    top_fps = select_top_k_events(fp_indices, scores, k=10, mode='max')
    
    print(f"\n🔎 绘制 Top 10 误报 (FP)...")
    for i, idx in enumerate(top_fps):
        plot_integrated_diagnosis(mat_orig, mat_recon, mat_pred, scores, labels, 
                             threshold=best_thresh,
                             start_idx=max(0, idx - 250), 
                             title=f"FP_Rank{i+1}")

    # B. 分析漏报 (FN)
    fn_indices = np.where((labels == 1) & (scores <= best_thresh))[0]
    top_fns = select_top_k_events(fn_indices, scores, k=10, mode='min')
    
    print(f"\n🔎 绘制 Top 10 漏报 (FN)...")
    for i, idx in enumerate(top_fns):
        plot_integrated_diagnosis(mat_orig, mat_recon, mat_pred, scores, labels, 
                             threshold=best_thresh,
                             start_idx=max(0, idx - 250), 
                             title=f"FN_Rank{i+1}")

    print(f"\n✅ 图片已保存至 {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()