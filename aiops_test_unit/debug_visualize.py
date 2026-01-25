import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

# === 配置 ===
DATA_DIR = 'data/final_dataset' 
GT_FILE = 'data/Aiops-Dataset/groundtruth/groundtruth-2022-05-01.csv' 
TARGET_SERVICE = 'frontend-0' 
SAVE_IMG_NAME = 'debug_alignment_zoom.png'

def check_alignment():
    print(f"🔍 [Zoom Mode] 正在深度检查服务: {TARGET_SERVICE}")

    # 1. 加载标签
    if not os.path.exists(GT_FILE): return
    df_gt = pd.read_csv(GT_FILE)
    df_gt = df_gt[df_gt['cmdb_id'] == TARGET_SERVICE]
    fault_times = df_gt['timestamp'].values
    if len(fault_times) == 0:
        print("⚠️ 该服务无故障记录")
        return
    
    # 取第一个故障时间作为观察点
    focus_time = fault_times[0]
    print(f"🎯 聚焦故障时间点: {focus_time}")

    # 2. 加载数据 (添加 weights_only=False 修复报错)
    search_path = os.path.join(DATA_DIR, f"{TARGET_SERVICE}*.pt")
    files = glob.glob(search_path)
    if not files: return
    
    data = torch.load(files[0], weights_only=False) # <--- 修复点
    # 智能判断：如果是 Tensor 就转，如果是 Numpy 就直接用
    timestamps = data['timestamps']
    if hasattr(timestamps, 'numpy'):
        timestamps = timestamps.numpy()
        
    metrics = data['metrics']
    if hasattr(metrics, 'numpy'):
        metrics = metrics.numpy()

    # 3. 截取故障前后 2小时 (7200秒) 的数据
    # 找到 focus_time 在 timestamps 里的索引
    # 使用 searchsorted 快速查找最近的时间点
    idx = np.searchsorted(timestamps, focus_time)
    
    # 定义窗口范围
    window = 120 # 前后 120 个点 (约20-30分钟，取决于采样率)
    start_idx = max(0, idx - window)
    end_idx = min(len(timestamps), idx + window)
    
    if start_idx >= end_idx:
        print("❌ 故障时间不在当前数据文件的时间范围内！")
        print(f"   数据范围: {timestamps[0]} ~ {timestamps[-1]}")
        return

    # 截取数据
    t_zoom = timestamps[start_idx:end_idx]
    m_zoom = metrics[start_idx:end_idx, :]

    print(f"📊 正在绘制局部放大图 ({t_zoom[0]} - {t_zoom[-1]})...")

    # 4. 画图
    plt.figure(figsize=(15, 8))
    
    # 子图1: 正常视角的 Metric Dim 0 (通常是 CPU/Latency)
    plt.subplot(2, 1, 1)
    plt.plot(t_zoom, m_zoom[:, 0], label='Metric Dim 0 (CPU/Lat)', color='blue')
    plt.axvline(x=focus_time, color='red', linestyle='--', linewidth=2, label='GT Label')
    # 尝试画一下 "UTC+8" 的位置，看看是否对齐
    plt.axvline(x=focus_time - 28800, color='orange', linestyle=':', linewidth=2, label='GT - 8h')
    plt.axvline(x=focus_time + 28800, color='purple', linestyle=':', linewidth=2, label='GT + 8h')
    
    plt.title(f"Zoomed View: {TARGET_SERVICE} (Dim 0)\nCenter=GT Timestamp", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 限制 Y 轴视角的 Metric Dim 1 & 2 (查看是否有微小波动)
    plt.subplot(2, 1, 2)
    plt.plot(t_zoom, m_zoom[:, 1], label='Metric Dim 1', alpha=0.7)
    plt.plot(t_zoom, m_zoom[:, 2], label='Metric Dim 2', alpha=0.7)
    plt.axvline(x=focus_time, color='red', linestyle='--', linewidth=2, label='GT Label')
    
    # 【关键】强制限制 Y 轴，过滤掉 35000 这种极端值
    plt.ylim(-10, 10) 
    plt.title("Zoomed View: Dim 1 & 2 (Y-axis Limited to -10~10)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(SAVE_IMG_NAME)
    print(f"✅ 生成新图片: {SAVE_IMG_NAME}")
    print("👉 请查看新图：\n1. 红色虚线周围，蓝线有没有波动？\n2. 或者是紫色/橙色虚线周围有波动？")

if __name__ == "__main__":
    check_alignment()