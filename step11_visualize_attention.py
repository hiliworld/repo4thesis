import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from step4_windowing import SMDWindowDataset
from model_v2_with_gat import MyFinalModel

# === 配置 ===
WINDOW_SIZE = 100
FEATURE_DIM = 36
MODEL_PATH = "my_trained_model_adaptive.pth" # 加载那个最好的模型
# 选一个表现最好的文件来画图，比如 machine-2-8
TEST_FILE = "data/ServerMachineDataset/test/machine-2-8.txt" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. 加载模型 ===
model = MyFinalModel(num_features=FEATURE_DIM, window_size=WINDOW_SIZE, hidden_dim=64, z_dim=16).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === 2. 加载一条数据 ===
# 我们只取其中的某一个时间窗口来观察
dataset = SMDWindowDataset(TEST_FILE, window_size=WINDOW_SIZE)
# 假设我们想看第 500 个时间窗 (你可以随便改，最好找个有故障的时刻)
sample_idx = 500 
x = dataset[sample_idx].unsqueeze(0).to(DEVICE) # [1, 100, 36]

# === 3. 推理并获取 Attention ===
with torch.no_grad():
    # 注意：我们的 forward 返回三个值：pred, recon, attn_weights
    _, _, attn_weights = model(x)
    
    # attn_weights shape: [1, 36, 36] (因为我们在模型里已经做过 mean(dim=1) 了吗？)
    # 让我们检查一下 adaptive_gat.py，
    # 里面返回的是 avg_attn_weights = attn_weights.mean(dim=1) -> [Batch, N, N]
    # 所以这里拿到的就是 [1, 36, 36]
    
    heatmap_data = attn_weights[0].cpu().numpy() # [36, 36]

# === 4. 画图 ===
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap="viridis", square=True)
plt.title(f"Adaptive Graph Attention Map (Sample {sample_idx})")
plt.xlabel("Source Node (Sensor)")
plt.ylabel("Target Node (Sensor)")
plt.savefig("attention_heatmap.png", dpi=300)
print("🖼️ 热力图已保存至 attention_heatmap.png")
plt.show()