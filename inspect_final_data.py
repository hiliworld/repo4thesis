import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

# === 配置 ===
DATA_DIR = "data/final_dataset"  # 你的 .pt 文件目录
SAMPLE_FILE_PREFIX = "frontend-0" # 重点检查这个服务（因为之前问题最大）

def check_data_quality():
    print(f"🩺 [Sanity Check] 正在检查目录: {DATA_DIR} ...")
    
    pt_files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    if not pt_files:
        print("❌ 错误：找不到 .pt 文件！请检查路径。")
        return

    # 1. 全局统计
    total_samples = 0
    nan_count = 0
    
    # 用来记录所有数据的最大最小值，看是否还有 35000
    global_max = -float('inf')
    global_min = float('inf')
    
    print(f"📊 正在扫描 {len(pt_files)} 个文件...")
    
    for f in tqdm(pt_files):
        try:
            data = torch.load(f, weights_only=False) # 兼容 PyTorch 新版安全策略
            
            # 提取数据
            metrics = data['metrics'] # [N, 333]
            
            # --- 检查 A: NaNs ---
            if torch.isnan(metrics).any():
                print(f"⚠️ 警告：文件 {os.path.basename(f)} 包含 NaN！")
                nan_count += 1
            
            # --- 检查 B: 极值 (Clip 效果) ---
            curr_max = metrics.max().item()
            curr_min = metrics.min().item()
            
            if curr_max > global_max: global_max = curr_max
            if curr_min < global_min: global_min = curr_min
            
            total_samples += metrics.shape[0]
            
        except Exception as e:
            print(f"❌ 读取错误 {os.path.basename(f)}: {e}")

    print("\n" + "="*40)
    print("📈 全局体检报告 (Global Health Report)")
    print("="*40)
    print(f"1. 总样本数: {total_samples}")
    print(f"2. 包含 NaN 的文件数: {nan_count} (目标: 0)")
    print(f"3. 全局最大值: {global_max:.4f} (目标: < 5.0，绝对不能是 35000!)")
    print(f"4. 全局最小值: {global_min:.4f} (目标: > -5.0)")
    
    if global_max > 10:
        print("🚨 [严重警告] 最大值依然很大！3σ Clipping 可能未生效！")
    elif global_max < 5:
        print("✅ [通过] 数值范围正常，StandardScaler 和 Clipping 工作良好。")
        
    # 2. 深度可视化 (Deep Dive Visualization)
    # 专门画一下 frontend-0 的前几个维度，确保波形还在，不是死线
    visualize_sample(SAMPLE_FILE_PREFIX)

def visualize_sample(prefix):
    print(f"\n🔬 [Deep Dive] 正在深度可视化: {prefix} ...")
    
    # 找对应的 train 文件
    search_path = os.path.join(DATA_DIR, f"*{prefix}*_train.pt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"❌ 找不到包含 {prefix} 的文件")
        return
        
    file_path = files[0]
    data = torch.load(file_path, weights_only=False)
    metrics = data['metrics'].numpy()
    
    # 画前 3 个维度的分布直方图 + 波形图
    plt.figure(figsize=(15, 10))
    
    # --- 子图 1: 数值分布直方图 ---
    plt.subplot(2, 2, 1)
    plt.hist(metrics.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title("Value Distribution (All Dimensions)\nShould be Bell Curve (Normal Dist)")
    plt.xlabel("Sigma (Standard Deviation)")
    plt.ylabel("Count")
    plt.xlim(-5, 5) # 聚焦看核心区域
    
    # --- 子图 2: 维度 0 的波形 ---
    plt.subplot(2, 2, 2)
    plt.plot(metrics[:1000, 0], alpha=0.8, label="Dim 0") # 只画前1000个点
    plt.title(f"Waveform (Dim 0) - First 1000 points\nShould show fluctuation, NOT flat line")
    plt.ylim(-4, 4)
    plt.legend()
    
    # --- 子图 3: 维度 1 的波形 ---
    plt.subplot(2, 2, 3)
    plt.plot(metrics[:1000, 1], alpha=0.8, color='orange', label="Dim 1")
    plt.title(f"Waveform (Dim 1)")
    plt.ylim(-4, 4)
    plt.legend()
    
    # --- 子图 4: 维度 100 (随机抽查) ---
    plt.subplot(2, 2, 4)
    plt.plot(metrics[:1000, 100], alpha=0.8, color='green', label="Dim 100")
    plt.title(f"Waveform (Dim 100)")
    plt.ylim(-4, 4)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("final_data_inspection.png")
    print("✅ 可视化完成！请查看生成图片: final_data_inspection.png")

if __name__ == "__main__":
    check_data_quality()