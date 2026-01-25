import torch
import torch.nn as nn
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# 引入我们之前的模块
from step15_hdfs_dataset import get_hdfs_loaders
from model_v2_with_gat import MyFinalModel

# === 配置 ===
CONFIG_FILE = 'config.yaml'
MODEL_PATH = 'hdfs_model.pth'
BATCH_SIZE = 128 # 测试可以大一点
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"🔥 设备: {DEVICE} | 开始 HDFS 日志模型评估")
    
    # 1. 加载配置和数据
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制覆盖为 log 模式
    config['dataset']['modality'] = 'log'
    
    # 获取测试集 (注意：这里会自动划分出测试集，并且包含 labels)
    _, test_loader, vocab_size = get_hdfs_loaders(batch_size=BATCH_SIZE)
    config['dataset']['vocab_size'] = vocab_size

    # 2. 加载模型
    model = MyFinalModel(config).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 模型权重加载成功！")
    except FileNotFoundError:
        print("❌ 错误：找不到模型文件。请先运行 step16 进行训练！")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none') # none 表示不取平均，我们要保留每个样本的 Loss

    # 3. 推理 (Inference)
    all_losses = []
    all_labels = []
    
    print("🚀 正在计算测试集异常分数...")
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(DEVICE)
            # y 是 Ground Truth (0=Normal, 1=Anomaly)
            
            # 这里的逻辑和训练一样：Next Token Prediction
            input_seq = x[:, :-1]
            target_token = x[:, -1]
            
            # 前向传播
            logits, _ = model(input_seq)
            
            # 计算 Loss (作为异常分数)
            # Loss 越大 -> 模型越惊讶 -> 越可能是异常
            loss = criterion(logits, target_token)
            
            all_losses.extend(loss.cpu().numpy())
            all_labels.extend(y.numpy())

    all_losses = np.array(all_losses)
    all_labels = np.array(all_labels)

    # 4. 评估指标 (Metrics)
    # AUC 是衡量无监督异常检测最好的指标之一，因为它不依赖阈值
    auc = roc_auc_score(all_labels, all_losses)
    print(f"\n📊 [Result] AUROC: {auc:.4f}")

    # 5. 寻找最佳阈值 (Best F1)
    # 既然我们要写论文，就得算出具体的 F1
    print("🔍 正在搜索最佳阈值...")
    best_f1 = 0
    best_thres = 0
    
    # 在分数范围内扫描 100 个点
    thresholds = np.linspace(all_losses.min(), all_losses.max(), 100)
    
    for th in thresholds:
        preds = (all_losses > th).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thres = th

    # 6. 最终报告
    print("="*50)
    print(f"🏆 最佳 F1-Score: {best_f1:.4f}")
    print(f"🎯 最佳阈值: {best_thres:.4f}")
    
    # 使用最佳阈值再算一遍详细指标
    final_preds = (all_losses > best_thres).astype(int)
    cm = confusion_matrix(all_labels, final_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"📌 混淆矩阵:\n [TN={tn}, FP={fp}]\n [FN={fn}, TP={tp}]")
    print(f"   Precision: {tp / (tp+fp+1e-8):.4f}")
    print(f"   Recall:    {tp / (tp+fn+1e-8):.4f}")
    print("="*50)

    # 7. (可选) 保存结果用于画图
    df_res = pd.DataFrame({'score': all_losses, 'label': all_labels})
    df_res.to_csv("hdfs_test_results.csv", index=False)
    print("💾 结果已保存至 hdfs_test_results.csv")

if __name__ == "__main__":
    evaluate()