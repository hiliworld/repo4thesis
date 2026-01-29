import torch
import numpy as np
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import get_dataloaders
from src.models.anomaly_model import MyFinalModel

# === 配置区 ===
CONFIG_FILE = "config.yaml"
MODEL_NAME = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📍 这里填入你那张图中漏报区域的大致起始点
# 根据你的截图文件名/X轴，应该是 149925 附近
TARGET_START_IDX = 149900 
ANALYSIS_LENGTH = 500  # 分析 500 个点

def analyze_root_cause():
    print(f"🔥 Micro-Diagnosis Tool | Focus Window: {TARGET_START_IDX} - {TARGET_START_IDX + ANALYSIS_LENGTH}")
    
    # 1. 加载资源
    try:
        _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
        with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
        config['dataset']['input_dim'] = feature_dim
        
        model = MyFinalModel(config).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 2. 定位数据 (快速跳过前面的数据)
    print("🚀 Seeking to target window...")
    target_data = []
    
    # 我们需要找到对应 index 的 batch。这比较笨，但最准确。
    current_idx = 0
    with torch.no_grad():
        for x in test_loader:
            batch_size = x.shape[0]
            
            # 如果这个 batch 包含了我们的目标区间
            if current_idx + batch_size > TARGET_START_IDX:
                # 计算 batch 内的偏移量
                start_in_batch = max(0, TARGET_START_IDX - current_idx)
                end_in_batch = min(batch_size, TARGET_START_IDX + ANALYSIS_LENGTH - current_idx)
                
                if start_in_batch < end_in_batch:
                    # 截取数据
                    x_slice = x[start_in_batch : end_in_batch].to(DEVICE)
                    
                    # 推理
                    pred, recon, z_comb= model(x_slice)
                    target = x_slice[:, -1, :]
                    recon_last = recon[:, -1, :]
                    
                    # 计算每个特征的误差 [Batch, Features]
                    # 不做 mean，我们要看每个特征
                    loss_matrix = (pred - target)**2 + (recon_last - target)**2
                    target_data.append(loss_matrix.cpu().numpy())
            
            current_idx += batch_size
            if current_idx >= TARGET_START_IDX + ANALYSIS_LENGTH:
                break
    
    if not target_data:
        print("❌ 未找到数据，请检查 TARGET_START_IDX 是否越界。")
        return

    # [Time, Features]
    error_matrix = np.concatenate(target_data, axis=0)
    
    # === 3. 核心分析：谁是罪魁祸首？ ===
    print("\n🕵️‍♂️ === 根因稳定性分析 (Root Cause Stability) ===")
    print("逻辑：如果Top-1特征一直不变，说明是'持续性故障'。如果乱跳，说明是'多重故障'。")
    print("-" * 60)
    print(f"{'Time Step':<15} | {'Top-1 Error Feat':<20} | {'Top-2 Error Feat':<20} | {'Total Score':<10}")
    print("-" * 60)
    
    # 采样打印 (防止刷屏，每隔 10 个点打印一次)
    for t in range(0, len(error_matrix), 10):
        losses = error_matrix[t]
        total_score = np.mean(losses)
        
        # 获取误差最大的前 3 个特征的索引
        top_indices = np.argsort(losses)[::-1][:3]
        
        feat1 = f"Feat {top_indices[0]} ({losses[top_indices[0]]:.4f})"
        feat2 = f"Feat {top_indices[1]} ({losses[top_indices[1]]:.4f})"
        
        abs_time = TARGET_START_IDX + t
        print(f"{abs_time:<15} | {feat1:<20} | {feat2:<20} | {total_score:.4f}")

    # === 4. 可视化：贡献度排名图 (Rank Plot) ===
    # 画一张图：X轴是时间，Y轴是 Top-1 特征的 ID
    # 如果是一条直线，说明故障源很稳定
    top1_features = np.argmax(error_matrix, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(TARGET_START_IDX, TARGET_START_IDX + len(top1_features)), top1_features, 
                c='red', s=10, alpha=0.6, label='Top-1 Error Feature')
    plt.yticks(range(0, feature_dim, 2)) # Y轴显示特征ID
    plt.grid(True, alpha=0.3)
    plt.title(f"Root Cause Evolution (Time {TARGET_START_IDX} - {TARGET_START_IDX + ANALYSIS_LENGTH})")
    plt.ylabel("Feature Index")
    plt.xlabel("Time Steps")
    plt.legend()
    
    save_path = "root_cause_analysis.png"
    plt.savefig(save_path)
    print(f"\n📊 根因演变图已保存至: {save_path}")
    print("👉 看图技巧：")
    print("   - 如果红点连成一条直线：说明虽然分数低，但模型一直盯着同一个特征报错（持续性故障）。")
    print("   - 如果红点上下乱跳：说明故障在不同指标间转移（级联故障）。")

if __name__ == "__main__":
    analyze_root_cause()