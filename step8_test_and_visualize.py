import torch
import torch.nn as nn
import numpy as np
import os
import yaml
from tqdm import tqdm
# from sklearn.metrics import roc_auc_score # 暂时不需要，因为还没 Label

# === 导入自定义模块 ===
try:
    from data_factory import get_dataloaders
    from model_v2_with_gat import MyFinalModel
except ImportError:
    print("❌ 错误：找不到自定义模块。")
    exit()

# === 配置参数 ===
MODEL_NAME = "my_trained_model_adaptive.pth"
CONFIG_FILE = "config.yaml"

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前计算设备: {DEVICE}")

# ==========================================
# 1. 准备数据
# ==========================================
print("📂 正在通过工厂加载测试数据...")
try:
    # get_dataloaders 返回: train_loader, val_loader, feature_dim
    # 我们这里只需要 test_loader (即 val_loader) 和 维度
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    
    print(f"✅ 数据加载成功！特征维度: {feature_dim}")
    print(f"   测试集 Batch 数: {len(test_loader)}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit()

# ==========================================
# 2. 加载模型
# ==========================================
# 读取原始配置
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# 【核心修复】：更新 config 中的 input_dim 为实际加载到的维度 (37)
# 这样模型初始化时就能拿到正确的维度，而不是默认的 38
config['dataset']['input_dim'] = feature_dim
config['dataset']['modality'] = 'metric' 

print(f"🤖 正在加载模型: {MODEL_NAME} ...")

try:
    # 【核心修复】：直接传入 config 字典，而不是分散的参数
    model = MyFinalModel(config).to(DEVICE)
    
    # 加载权重
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    print("✅ 模型权重加载成功！")
except TypeError as e:
    print(f"❌ 模型初始化参数错误: {e}")
    exit()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

model.eval()

# ==========================================
# 3. 全量推理 (Inference)
# ==========================================
all_scores = []
criterion = nn.MSELoss(reduction='none')

print("🚀 开始推理...")
with torch.no_grad():
    for x in tqdm(test_loader):
        x = x.to(DEVICE)
        
        # 前向传播
        pred_next, recon_window, _ = model(x)
        
        # === 计算异常分 ===
        # 1. 预测误差 (只看最后一个点)
        target_next = x[:, -1, :]
        loss_pred = torch.mean((pred_next - target_next) ** 2, dim=1)
        
        # 2. 重建误差 (整个窗口取平均)
        loss_recon = torch.mean((recon_window - x) ** 2, dim=(1, 2))
        
        # 3. 综合得分
        score = loss_pred + loss_recon
        
        all_scores.append(score.cpu().numpy())

all_scores = np.concatenate(all_scores)

# ==========================================
# 4. 结果分析
# ==========================================
print("\n" + "="*50)
print("📊 测试概览")
print("="*50)
print(f"测试样本总数: {len(all_scores)}")
print(f"异常分范围: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
print(f"异常分均值: {np.mean(all_scores):.4f}")

# 画图
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.hist(all_scores, bins=50, color='blue', alpha=0.7)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.savefig("test_score_dist.png")
    print("✅ 分布图已保存至 test_score_dist.png")
except:
    pass