import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# === 导入自定义模块 ===
try:
    from step4_windowing import SMDWindowDataset
    from model_v2_with_gat import MyFinalModel
    from sklearn.metrics import precision_recall_curve, roc_auc_score
except ImportError:
    print("❌ 错误：找不到自定义模块，请检查文件名。")
    exit()

# === 配置参数 ===
WINDOW_SIZE = 100
BATCH_SIZE = 256  # 推理时不反向传播，Batch 可以大一点，速度快
FEATURE_DIM = 36
# 注意：这里要加载我们在 Step 7 训练出来的最新模型 (带对比学习的)
MODEL_NAME = "my_trained_model_cl.pth" 

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"🔥 当前计算设备: {DEVICE}")

# ==========================================
# 1. 准备路径
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(base_dir, 'data', 'ServerMachineDataset', 'test')
test_label_dir = os.path.join(base_dir, 'data', 'ServerMachineDataset', 'test_label')
model_path = os.path.join(base_dir, MODEL_NAME)

# 获取所有测试文件列表
test_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.txt')])
print(f"📂 发现 {len(test_files)} 个测试文件，准备开始批量评估...")

# ==========================================
# 2. 加载模型
# ==========================================
if not os.path.exists(model_path):
    print(f"❌ 错误：找不到模型文件 {model_path}")
    print("   请确保你已经运行了 Step 7 并且模型保存名为 my_trained_model_cl.pth")
    exit()

print(f"🤖 正在加载模型: {MODEL_NAME} ...")
# 注意：这里必须和训练时的参数完全一致 (z_dim=16)
model = MyFinalModel(num_features=FEATURE_DIM, window_size=WINDOW_SIZE, hidden_dim=64, z_dim=16).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ==========================================
# 3. 定义单文件评估函数
# ==========================================
def evaluate_one_file(filename):
    file_path = os.path.join(test_data_dir, filename)
    label_path = os.path.join(test_label_dir, filename)
    
    # 1. 加载数据
    # 注意：Step 4 的 Dataset 类如果不支持传入 window_size，请确保它内部默认是 100
    dataset = SMDWindowDataset(file_path, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # 2. 加载标签
    try:
        raw_labels = pd.read_csv(label_path, header=None).values.flatten()
        # 对齐标签：因为 Window 切片会损失前 (Window-1) 个点
        # 我们的模型预测的是窗口的最后一个点
        labels = raw_labels[WINDOW_SIZE-1:]
    except:
        return None, None, "No Label"

    # 长度二次检查 (处理 DataLoader 可能存在的 drop_last 行为，虽然上面设了 False)
    # 这一步是为了防止维度对其报错
    if len(labels) == 0:
        return None, None, "Empty Label"

    # 3. 推理
    criterion = nn.MSELoss(reduction='none')
    anomaly_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(DEVICE)
            # x shape: [Batch, 100, 36]
            
            target_next = x[:, -1, :] # 真实未来
            target_window = x         # 真实窗口

            pred_next, recon_window, _ = model(x)
            
            # --- 计算双重得分 ---
            # Score 1: 预测误差 (下一时刻)
            loss_forecast = torch.mean(criterion(pred_next, target_next), dim=1)
            
            # Score 2: 重建误差 (窗口最后一个点)
            loss_recon = torch.mean(criterion(recon_window[:, -1, :], target_window[:, -1, :]), dim=1)
            
            # 综合得分
            total_score = loss_forecast + loss_recon
            anomaly_scores.extend(total_score.cpu().numpy())
            
    anomaly_scores = np.array(anomaly_scores)
    
    # 截断对齐 (防止数据加载器和标签长度有细微差别)
    min_len = min(len(labels), len(anomaly_scores))
    anomaly_scores = anomaly_scores[:min_len]
    labels = labels[:min_len]
    
    return labels, anomaly_scores, "OK"

# ==========================================
# 4. 主循环：批量测试
# ==========================================
results = []
print("\n🚀 开始全量测试 (ProgressBar)...")

for filename in tqdm(test_files):
    labels, scores, status = evaluate_one_file(filename)
    
    if status != "OK":
        print(f"⚠️ 跳过 {filename}: {status}")
        continue
        
    # 如果该文件全是 0 (没有异常)，无法计算 AUC，跳过
    if np.sum(labels) == 0:
        # print(f"ℹ️ {filename} 没有异常样本，跳过指标计算")
        continue
        
    # 计算指标
    try:
        # AUC
        auc = roc_auc_score(labels, scores)
        
        # Best F1 (Pot-eval 策略)
        prec, rec, _ = precision_recall_curve(labels, scores)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_f1 = np.max(f1_scores)
        
        results.append({
            "File": filename,
            "AUC": auc,
            "Best_F1": best_f1
        })
    except Exception as e:
        print(f"❌ 计算指标出错 {filename}: {e}")

# ==========================================
# 5. 最终报告
# ==========================================
if len(results) > 0:
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("📊 最终测试报告 (Server Machine Dataset)")
    print("="*40)
    print(f"测试机器数量: {len(df_res)}")
    print(f"平均 AUC     : {df_res['AUC'].mean():.4f}")
    print(f"平均 Best F1 : {df_res['Best_F1'].mean():.4f}")
    print("-" * 40)
    print("表现最好的 3 个机器:")
    print(df_res.sort_values(by="Best_F1", ascending=False).head(3))
    print("-" * 40)
    
    # 保存详细结果到 CSV，方便写论文画表
    csv_path = os.path.join(base_dir, "final_test_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"📝 详细结果已保存至: {csv_path}")
    
else:
    print("❌ 没有得到任何有效结果，请检查数据路径或标签文件。")