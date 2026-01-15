import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# 引入 f1_score 计算
from sklearn.metrics import roc_auc_score, f1_score

# === 导入自定义模块 ===
try:
    from step4_windowing import SMDWindowDataset
    from model_v2_with_gat import MyFinalModel
except ImportError:
    print("❌ 错误：找不到自定义模块，请检查文件名。")
    exit()

# === 配置参数 ===
WINDOW_SIZE = 100
BATCH_SIZE = 256
FEATURE_DIM = 36
# ✅ 确保加载的是完全体模型
MODEL_NAME = "my_trained_model_adaptive.pth" 

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

test_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.txt')])
print(f"📂 发现 {len(test_files)} 个测试文件。")

# ==========================================
# 2. 加载模型
# ==========================================
if not os.path.exists(model_path):
    print(f"❌ 错误：找不到模型文件 {model_path}")
    print("   请确保 Step 9 训练完成并保存了模型。")
    exit()

print(f"🤖 正在加载模型: {MODEL_NAME} ...")
model = MyFinalModel(num_features=FEATURE_DIM, window_size=WINDOW_SIZE, hidden_dim=64, z_dim=16).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ==========================================
# ✅ 核心函数：Point Adjustment (PA)
# ==========================================
def point_adjustment(preds, labels):
    """
    PA 策略实现：
    如果模型在一个连续的异常片段中正确检测到了哪怕 1 个点，
    我们就把这整个片段的所有点的预测结果都置为 1 (视为检测成功)。
    """
    adjusted_preds = preds.copy()
    if np.sum(labels) == 0:
        return adjusted_preds
    
    # 找到所有异常片段的起止点
    # diff 为 1 的位置是开始，-1 的位置是结束
    diff = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        # 只要该片段内有一个点预测为 1
        if np.sum(preds[start:end]) > 0:
            adjusted_preds[start:end] = 1
            
    return adjusted_preds

# ==========================================
# ✅ 核心函数：搜索最佳 F1 (PA-F1)
# ==========================================
def get_best_f1_with_pa(scores, labels, step_num=100):
    """
    遍历可能的阈值，应用 PA，找到最高的 F1 分数
    """
    # 生成 100 个候选阈值 (从 0% 到 100% 分位数)
    # 这样比遍历所有分数要快得多，精度也足够
    min_score, max_score = np.min(scores), np.max(scores)
    # 稍微放宽一点边界
    thresholds = np.linspace(min_score, max_score, step_num)
    
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    # 遍历阈值
    for th in thresholds:
        # 1. 生成原始二分类预测
        preds = (scores > th).astype(int)
        
        # 2. 如果预测全是0，跳过 (防止除0错误)
        if np.sum(preds) == 0:
            continue
            
        # 3. 应用 Point Adjustment
        adjusted_preds = point_adjustment(preds, labels)
        
        # 4. 计算 F1
        f1 = f1_score(labels, adjusted_preds)
        
        if f1 > best_f1:
            best_f1 = f1
            # 顺便记录下此时的 P 和 R，写论文可能要用
            # best_precision = precision_score(labels, adjusted_preds)
            # best_recall = recall_score(labels, adjusted_preds)
            
    return best_f1

# ==========================================
# 3. 单文件评估逻辑
# ==========================================
def evaluate_one_file(filename):
    file_path = os.path.join(test_data_dir, filename)
    label_path = os.path.join(test_label_dir, filename)
    
    dataset = SMDWindowDataset(file_path, window_size=WINDOW_SIZE)
    # drop_last=False 保证测试数据不丢失
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    try:
        raw_labels = pd.read_csv(label_path, header=None).values.flatten()
        labels = raw_labels[WINDOW_SIZE-1:]
    except:
        return None, None, "No Label"

    if len(labels) == 0:
        return None, None, "Empty Label"

    criterion = nn.MSELoss(reduction='none')
    anomaly_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(DEVICE)
            target_next = x[:, -1, :] 
            target_window = x         

            pred_next, recon_window, _ = model(x)
            
            loss_forecast = torch.mean(criterion(pred_next, target_next), dim=1)
            loss_recon = torch.mean(criterion(recon_window[:, -1, :], target_window[:, -1, :]), dim=1)
            
            total_score = loss_forecast + loss_recon
            anomaly_scores.extend(total_score.cpu().numpy())
            
    anomaly_scores = np.array(anomaly_scores)
    
    # 对齐
    min_len = min(len(labels), len(anomaly_scores))
    anomaly_scores = anomaly_scores[:min_len]
    labels = labels[:min_len]
    
    return labels, anomaly_scores, "OK"

# ==========================================
# 4. 主循环
# ==========================================
results = []
print("\n🚀 开始全量测试 (含 Point Adjustment)...")

for filename in tqdm(test_files):
    labels, scores, status = evaluate_one_file(filename)
    
    if status != "OK":
        continue
        
    if np.sum(labels) == 0:
        continue
        
    try:
        # 计算 AUC (AUC 不受 PA 影响，直接算)
        auc = roc_auc_score(labels, scores)
        
        # 计算 PA-F1 (这是改动的核心)
        best_f1_pa = get_best_f1_with_pa(scores, labels)
        
        results.append({
            "File": filename,
            "AUC": auc,
            "Best_F1_PA": best_f1_pa  # 标记为 PA 版本
        })
    except Exception as e:
        print(f"❌ 计算指标出错 {filename}: {e}")

# ==========================================
# 5. 输出报告
# ==========================================
if len(results) > 0:
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("📊 最终测试报告 (Point Adjusted)")
    print("="*50)
    print(f"测试机器数量: {len(df_res)}")
    print(f"平均 AUC        : {df_res['AUC'].mean():.4f}")
    print(f"平均 Best F1 (PA): {df_res['Best_F1_PA'].mean():.4f}")
    print("-" * 50)
    print("表现最好的 3 个机器:")
    print(df_res.sort_values(by="Best_F1_PA", ascending=False).head(3))
    print("-" * 50)
    
    csv_path = os.path.join(base_dir, "final_test_results_pa.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"📝 结果已保存: {csv_path}")
    
else:
    print("❌ 无有效结果。")