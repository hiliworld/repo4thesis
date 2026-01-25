import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

# === 配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data/final_dataset")
GT_FILE = os.path.join(BASE_DIR, "data/Aiops-Dataset/groundtruth/groundtruth-2022-05-01.csv")

TARGET_SERVICE = "frontend-0"
# 根据你的 RF 结果，Dim 4, 6 是最重要的，我们重点看它
TOP_FEATURES = [4, 6, 8] 
FAULT_DURATION = 300 

def load_ground_truth():
    if not os.path.exists(GT_FILE):
        print(f"❌ 找不到标签文件: {GT_FILE}")
        sys.exit(1)
    return pd.read_csv(GT_FILE)

def visualize_alignment():
    print(f"🕵️‍♂️ [Alignment Check] 正在深度检查服务: {TARGET_SERVICE} ...")
    
    gt_df = load_ground_truth()
    faults = gt_df[gt_df['cmdb_id'] == TARGET_SERVICE]['timestamp'].values
    print(f"   -> 发现 GT 故障时刻: {faults}")
    
    if len(faults) == 0:
        print("❌ 该服务无故障记录")
        return

    # 我们取第一个故障时间来定位
    target_fault_time = faults[0]
    
    # === 智能搜索：找包含这个时间点的文件 ===
    search_pattern = os.path.join(DATA_DIR, f"{TARGET_SERVICE}*.pt")
    all_files = glob.glob(search_pattern)
    
    target_file = None
    data_metrics = None
    data_timestamps = None
    data_logs = None
    
    print(f"   -> 正在扫描 {len(all_files)} 个文件，寻找包含时间戳 {target_fault_time} 的数据...")
    
    for f in all_files:
        try:
            # 只读取 timestamps 头部和尾部来快速判断（或者是轻量级读取）
            # 为了准确，我们还是得读进来，但因为 .pt 很快，所以还好
            temp_data = torch.load(f, weights_only=False)
            ts = temp_data['timestamps']
            
            t_min, t_max = ts.min(), ts.max()
            
            # 检查故障时间是否在这个文件的时间范围内
            if t_min <= target_fault_time <= t_max:
                target_file = f
                data_metrics = temp_data['metrics'].numpy()
                data_timestamps = ts
                data_logs = temp_data['logs']
                print(f"✅ 找到了！故障在文件: {os.path.basename(f)}")
                print(f"   (时间范围: {t_min:.0f} ~ {t_max:.0f})")
                break
        except:
            continue
            
    if target_file is None:
        print("❌ 悲剧了：所有文件中都找不到覆盖该故障时间段的数据。")
        print("   -> 可能原因：数据清洗时该时间段被切掉了（比如刚好在文件头尾被截断）。")
        return

    # === 开始绘图 ===
    log_counts = np.array([len(seq) for seq in data_logs])
    
    # 聚焦窗口: 故障前 10分钟 ~ 故障后 15分钟
    start_time = target_fault_time - 600
    end_time = target_fault_time + 900
    
    mask = (data_timestamps >= start_time) & (data_timestamps <= end_time)
    plot_indices = np.where(mask)[0]
    
    if len(plot_indices) == 0:
        print("⚠️ 奇怪，文件时间范围匹配，但切片为空。画全部数据。")
        plot_indices = np.arange(len(data_timestamps))

    plt.figure(figsize=(15, 12))
    
    # 1. Metrics
    plt.subplot(3, 1, 1)
    for dim in TOP_FEATURES:
        plt.plot(data_timestamps[plot_indices], data_metrics[plot_indices, dim], label=f"Feat Dim {dim}", linewidth=1.5, alpha=0.9)
    
    # 画红区
    plt.axvspan(target_fault_time, target_fault_time + FAULT_DURATION, color='red', alpha=0.3, label="GT Anomaly Window")
    plt.title(f"Metric Alignment: Do metrics explode in the Red Zone? ({TARGET_SERVICE})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 2. Logs
    plt.subplot(3, 1, 2)
    plt.plot(data_timestamps[plot_indices], log_counts[plot_indices], color='purple', label="Log Volume", linewidth=1.5)
    plt.axvspan(target_fault_time, target_fault_time + FAULT_DURATION, color='red', alpha=0.3)
    plt.title("Log Alignment: Is there a log burst in the Red Zone?")
    plt.grid(True, alpha=0.3)
    
    # 3. Heatmap
    plt.subplot(3, 1, 3)
    subset = data_metrics[plot_indices].T
    # 选方差最大的 30 个维度
    var_idx = np.argsort(subset.var(axis=1))[::-1][:30]
    plt.imshow(subset[var_idx, :], aspect='auto', cmap='coolwarm', vmin=-5, vmax=5)
    plt.title("Heatmap of Top 30 Active Metrics")
    plt.colorbar(label="Robust Scaled Value")
    
    plt.tight_layout()
    plt.savefig("alignment_verify_v2.png")
    print("\n📸 最终验证图: alignment_verify_v2.png")

if __name__ == "__main__":
    visualize_alignment()