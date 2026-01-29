import torch
import numpy as np
import pandas as pd
import os
import yaml
import glob
import matplotlib
matplotlib.use('Agg') # 服务器绘图防卡死
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# === 导入你的模块 ===
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
OUTPUT_DIR = "fault_clusters_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
HARDCODED_LABEL_PATH = "/home/sde/MyThesis/data/ServerMachineDataset/test_label"

# === 1. 辅助函数：加载标签 ===
def load_labels_robust(config):
    """加载并拼接所有标签，确保与测试集长度对齐"""
    # === 修复逻辑：优先使用硬编码路径，防止路径替换出错 ===
    if os.path.exists(HARDCODED_LABEL_PATH):
        label_path = HARDCODED_LABEL_PATH
    else:
        # 如果硬编码路径不存在，再尝试智能推导
        test_path_conf = config['dataset']['test_file']
        # 只做一次替换，防止出现 test_label_label
        label_path = test_path_conf.replace("/test", "/test_label")
    
    pattern = config['dataset']['format']['pattern']
    search_path = os.path.join(label_path, pattern)
    files = sorted(glob.glob(search_path))
    
    label_list = []
    window = config['dataset']['window_size']
    
    print(f"📂 Loading labels from: {label_path}")
    if not files:
        print(f"❌ No label files found in {search_path}")
        return None

    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            raw = df.values.flatten()
            # 注意：Loader 切窗时会丢掉前 window_size 个点
            if len(raw) > window: 
                label_list.append(raw[window:])
        except: pass
        
    if not label_list: return None
    return np.concatenate(label_list)

# === 2. 核心逻辑：故障事件提取 ===
def extract_fault_events(model, loader, labels, device):
    """
    遍历数据，找到所有 label=1 的片段，提取其 Latent Vector 的均值作为该故障的'指纹'
    """
    print("🚀 Extracting latent features for all data...")
    latent_vectors = []
    
    # A. 获取所有时刻的隐变量 z
    with torch.no_grad():
        for x in tqdm(loader, desc="Inference"):
            x = x.to(device)
            # 兼容不同版本的模型返回
            ret = model(x)
            # 假设最后一个返回值通常是 z_combined 或 z_proj
            # 如果是 V2 (pred, recon, z_node, z_proj)，取 -1
            # 如果是 V1 (pred, recon, z_comb)，取 -1
            z = ret[-1] 
            
            # z shape: [Batch, Nodes, Hidden] -> Flatten -> [Batch, Nodes*Hidden]
            # 我们把所有节点的特征拼起来，代表整个机器的状态
            z_flat = z.view(z.shape[0], -1).cpu().numpy()
            latent_vectors.append(z_flat)
            
    all_latents = np.concatenate(latent_vectors, axis=0)
    
    # 确保长度对齐
    min_len = min(len(all_latents), len(labels))
    all_latents = all_latents[:min_len]
    labels = labels[:min_len]
    
    # B. 切分故障片段 (Event Segmentation)
    # 找到 label 连续为 1 的区间
    print("🔍 Grouping faults into events...")
    fault_events = [] # 存每个故障的特征向量
    event_indices = [] # 存每个故障的起止时间 (start, end)
    
    is_fault = False
    start_idx = 0
    
    for i in range(len(labels)):
        if labels[i] == 1 and not is_fault:
            is_fault = True
            start_idx = i
        elif labels[i] == 0 and is_fault:
            is_fault = False
            end_idx = i
            # 只有持续时间 > 10 的才算有效故障，过滤噪点
            if end_idx - start_idx > 10:
                # 核心：取这段时间内所有 z 的平均值，代表这个故障的"语义"
                event_rep = np.mean(all_latents[start_idx:end_idx], axis=0)
                fault_events.append(event_rep)
                event_indices.append((start_idx, end_idx))
                
    if is_fault: # 处理最后一段
        event_rep = np.mean(all_latents[start_idx:], axis=0)
        fault_events.append(event_rep)
        event_indices.append((start_idx, len(labels)))
        
    print(f"✅ Found {len(fault_events)} distinct fault events.")
    return np.array(fault_events), event_indices, all_latents

# === 3. 主程序 ===
def main():
    # 1. 准备环境
    try:
        _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    except Exception as e:
        print(f"❌ Dataloader Error: {e}")
        return

    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    config['dataset']['input_dim'] = feature_dim
    
    # 2. 加载模型
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    model.eval()
    
    # 3. 加载标签
    labels = load_labels_robust(config)
    if labels is None:
        print("❌ Labels not found.")
        return
        
    # 4. 提取故障特征
    fault_feats, event_indices, all_latents = extract_fault_events(model, test_loader, labels, DEVICE)
    
    if len(fault_feats) < 5:
        print("❌ 故障事件太少 (<5)，无法聚类。请检查标签路径或模型读取。")
        return

    # 5. K-Means 聚类
    # 假设我们想把故障分为 5 类 (你可以根据需要调整 K)
    K = 5
    print(f"🧩 Clustering {len(fault_feats)} events into {K} types...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(fault_feats)
    
    # 6. t-SNE 可视化 (把高维特征降到 2D 来看分布)
    print("🎨 Generating t-SNE plot...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(fault_feats)-1), random_state=42)
    feats_2d = tsne.fit_transform(fault_feats)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=cluster_labels, cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(scatter, label='Fault Type (Cluster ID)')
    plt.title(f"Fault Landscape: {len(fault_feats)} Events Clustered into {K} Types")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    
    # 标注每个点是哪个故障
    for i, txt in enumerate(cluster_labels):
        plt.annotate(str(txt), (feats_2d[i, 0], feats_2d[i, 1]), fontsize=8, alpha=0.7)
        
    save_path = f"{OUTPUT_DIR}/fault_clusters_tsne.png"
    plt.savefig(save_path)
    print(f"   📸 Saved map: {save_path}")
    
    # 7. 生成故障画像 (每个 Cluster 选一个代表画出来)
    # 我们没法画 Latent，但我们可以画那个时间段的 Top 异常特征的波形
    print("📊 Generating Cluster Profiles...")
    
    # 重新获取原始数据以便画图 (这里简化处理，只加载一部分)
    # 为了效率，我们只记录每个 Cluster 的第一个事件的 Index
    
    # 保存聚类结果 CSV
    df_res = pd.DataFrame({
        'Event_ID': range(len(cluster_labels)),
        'Start_Idx': [e[0] for e in event_indices],
        'End_Idx': [e[1] for e in event_indices],
        'Duration': [e[1]-e[0] for e in event_indices],
        'Cluster_Type': cluster_labels
    })
    csv_path = f"{OUTPUT_DIR}/fault_clustering_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"   📝 Saved cluster list: {csv_path}")
    
    # 打印统计
    print("\n=== 📊 故障分布统计 (Cluster Distribution) ===")
    print(df_res['Cluster_Type'].value_counts().sort_index())
    
    print("\n💡 下一步思路：")
    print(f"1. 打开 {csv_path}，查看每一类故障的时间段。")
    print("2. 挑选样本最多的 3-4 个类作为 '基类 (Base Classes)' (80% 训练数据)。")
    print("3. 挑选样本最少的 1-2 个类作为 '新类 (Novel Classes)' (20% 测试数据)。")
    print("4. 使用这些数据进行 Few-Shot Learning 实验。")

if __name__ == "__main__":
    main()