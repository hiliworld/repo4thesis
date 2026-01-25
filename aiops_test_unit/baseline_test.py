import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import glob
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 路径配置 (自动适配) ===
# 当前脚本所在目录: /home/sde/MyThesis/aiops_test_unit/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录: /home/sde/MyThesis/
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data/final_dataset")
GT_FILE = os.path.join(BASE_DIR, "data/Aiops-Dataset/groundtruth/groundtruth-2022-05-01.csv")

# === 参数配置 ===
FAULT_DURATION = 300     # 故障持续时间 (秒)
SAMPLE_RATE = 0.5        # 采样率
n_estimators = 50        # 树的数量

def load_ground_truth():
    """加载 GroundTruth CSV 文件"""
    if not os.path.exists(GT_FILE):
        print(f"❌ 错误：找不到标签文件: {GT_FILE}")
        sys.exit(1)
    return pd.read_csv(GT_FILE)

def generate_labels(timestamps, service_name, gt_df):
    """根据 GroundTruth 给时间戳打标"""
    labels = np.zeros(len(timestamps), dtype=int)
    # 筛选当前服务的故障
    service_faults = gt_df[gt_df['cmdb_id'] == service_name]['timestamp'].values
    
    for fault_time in service_faults:
        # 标记故障发生后 FAULT_DURATION 秒内的所有样本
        # 放宽一点判定范围 (-10s ~ +duration) 以防对齐微小偏差
        mask = (timestamps >= fault_time - 10) & (timestamps <= fault_time + FAULT_DURATION)
        labels[mask] = 1
    return labels

def load_all_data(gt_df):
    """加载目录下所有的 .pt 文件 (不分 train/test)"""
    # 搜索 data/final_dataset 下所有的 .pt 文件
    search_path = os.path.join(DATA_DIR, "*.pt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"❌ 未找到任何 .pt 文件！路径: {DATA_DIR}")
        return None, None

    print(f"📥 [All Data] 正在加载 {len(files)} 个文件 (采样率: {SAMPLE_RATE})...")
    
    X_list = []
    y_list = []
    
    for f in tqdm(files):
        try:
            filename = os.path.basename(f)
            # 兼容 _train.pt 和 _test.pt，提取核心服务名
            # frontend-0.source..._train.pt -> frontend-0
            service_name = filename.split('.')[0]
            
            data = torch.load(f, weights_only=False)
            metrics = data['metrics'].numpy()
            timestamps = data['timestamps']
            
            labels = generate_labels(timestamps, service_name, gt_df)
            
            # 降采样
            if SAMPLE_RATE < 1.0:
                indices = np.random.choice(len(metrics), int(len(metrics) * SAMPLE_RATE), replace=False)
                metrics = metrics[indices]
                labels = labels[indices]
            
            X_list.append(metrics)
            y_list.append(labels)
            
        except Exception as e:
            print(f"   ❌ 读取失败 {filename}: {e}")
            
    if not X_list: return None, None
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # 清理 NaN
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y

def run_baseline_test():
    print(f"🚀 [Baseline] 启动随机森林基准测试 (混合切分版)")
    
    gt_df = load_ground_truth()
    
    # 1. 加载所有数据
    X, y = load_all_data(gt_df)
    
    if X is None: return

    # 检查总异常数
    total_anomalies = np.sum(y == 1)
    print(f"📊 数据总量: {X.shape}, 总异常样本: {total_anomalies} ({total_anomalies/len(y):.4%})")

    if total_anomalies == 0:
        print("❌ [致命] 整个数据集中都没有异常样本！请检查 GroundTruth 时间戳是否正确。")
        return

    # 2. 混合切分 (Stratified Shuffle Split)
    # 这保证了训练集和测试集里都有相同比例的故障样本
    print("\n🔪 正在执行分层随机切分 (Train 80% / Test 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train异常数: {np.sum(y_train==1)} | Test异常数: {np.sum(y_test==1)}")

    # 3. 训练
    print(f"\n🌲 开始训练 RF (Trees={n_estimators})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 4. 评估
    print("\n🔮 正在预测...")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\n🌟 [最终结果] ROC-AUC: {auc:.4f}")
    except:
        print("\n⚠️ 无法计算 AUC (可能是测试集只有一个类别)")

    print("\n[分类报告]")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\n[混淆矩阵]")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. 特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), importances[indices])
    plt.xticks(range(10), indices)
    plt.title(f"RF Feature Importance (Top 10) - AUC: {auc:.4f}")
    plt.savefig("baseline_rf_result_stratified.png")
    print("\n📸 结果图已保存至: baseline_rf_result_stratified.png")

if __name__ == "__main__":
    run_baseline_test()