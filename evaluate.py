import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import glob
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

# === 导入自定义模块 ===
try:
    from data_factory import get_dataloaders
    from model_v2_with_gat import MyFinalModel
except ImportError:
    print("❌ 错误：找不到自定义模块。")
    exit()

# === 配置参数 ===
CONFIG_FILE = "config.yaml"
MODEL_NAME = "my_trained_model_adaptive.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_labels(config):
    """
    专门加载 SMD 的标签文件，并与测试集的时间窗口对齐
    """
    test_path = config['dataset']['test_file']
    # 标签通常在 test 目录同级的 test_label 目录
    # 例如: data/ServerMachineDataset/test/ -> data/ServerMachineDataset/test_label/
    label_path = test_path.replace("test", "test_label")
    
    if not os.path.exists(label_path):
        print(f"⚠️ 警告：找不到标签目录 {label_path}，无法计算 F1/AUC")
        return None

    # 获取文件列表 (必须排序，保证和 Dataset 读取顺序一致)
    pattern = config['dataset']['format'].get('pattern', '*.txt')
    label_files = sorted(glob.glob(os.path.join(label_path, pattern)))
    
    window_size = config['dataset']['window_size']
    aligned_labels = []
    
    print(f"🏷️ 正在加载标签 (共 {len(label_files)} 个文件)...")
    
    for lf in label_files:
        try:
            # 读取标签 (SMD标签文件通常没有表头，只有一列 0/1)
            df = pd.read_csv(lf, header=None)
            raw_labels = df.values.flatten() # 转成一维数组
            
            # === 关键对齐步骤 ===
            # SmartTimeSeriesDataset 切窗逻辑是: len(data) - window_size
            # 对应的 Label 应该是窗口的【最后一个点】
            # 所以 Label 应该从 index = window_size 开始取
            # 比如 window=100，第一个样本是 0-99，对应的 label 是 index 99 (第100个点)
            # 但 Dataset 实际上是 range(len - window)
            # 第 0 个窗口: data[0:100], target 是 data[100] (预测 Next) 或 data[0:100] (重建)
            # 通常异常检测以“当前时刻是否异常”为准。
            # 如果是预测下一个点，Label 对应 index 100。
            # 如果是重建窗口，Label 通常对应窗口末端。
            
            # 这里我们采用标准做法：Label 截取掉前 window_size 个点
            # 假设文件长 1000，window 100。
            # 生成 900 个窗口。
            # 对应 Label 是 raw_labels[100:] (长度 900)
            
            if len(raw_labels) > window_size:
                valid_labels = raw_labels[window_size:] 
                aligned_labels.append(valid_labels)
            
        except Exception as e:
            print(f"读取标签失败 {lf}: {e}")
            
    if len(aligned_labels) == 0:
        return None
        
    return np.concatenate(aligned_labels)

def get_best_f1(labels, scores):
    """
    动态搜索最佳阈值 (Best F1)
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_f1 = np.max(f1_scores)
    best_thresh = thresholds[np.argmax(f1_scores)]
    return best_f1, best_thresh

def point_adjustment(score, label, thres):
    """
    AIOps 标准：Point Adjustment (PA)
    如果在一段连续的异常区间内，只要有一个点被检测到 (score > thres)，
    则整个区间的预测值都被置为 1 (视为成功召回)。
    """
    predict = score > thres
    actual = label > 0.5
    anomaly_state = False
    anomaly_count = 0
    
    # 遍历所有点
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            # 进入异常区间，且检测到了
            anomaly_state = True
            # 向前回溯：把同属一个异常片段的漏报点全部修正为 1
            for j in range(i, 0, -1):
                if not actual[j]: break
                else:
                    if not predict[j]:
                        predict[j] = True
            
        elif not actual[i]:
            anomaly_state = False
            
        if anomaly_state:
            predict[i] = True
            
    return predict

def main():
    # 1. 加载配置
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. 准备数据和模型
    _, test_loader, feature_dim = get_dataloaders(CONFIG_FILE)
    config['dataset']['input_dim'] = feature_dim
    
    model = MyFinalModel(config).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
    model.eval()
    
    # 3. 推理 (Inference)
    print("🚀 正在重新推理以计算指标...")
    all_scores = []
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(DEVICE)
            pred_next, recon_window, _ = model(x)
            
            target_next = x[:, -1, :]
            # 预测误差
            l_pred = torch.mean((pred_next - target_next) ** 2, dim=1)
            # 重建误差
            l_recon = torch.mean((recon_window - x) ** 2, dim=(1, 2))
            
            score = l_pred + l_recon
            all_scores.append(score.cpu().numpy())
            
    y_pred_raw = np.concatenate(all_scores)
    
    # 4. 加载标签 (Ground Truth)
    y_true = load_labels(config)
    
    if y_true is None:
        print("❌ 无法加载标签，结束。")
        return

    # 5. 长度对齐检查
    # 因为 drop_last=True 或者 某些切窗边界问题，可能差几个点
    min_len = min(len(y_pred_raw), len(y_true))
    y_pred_raw = y_pred_raw[:min_len]
    y_true = y_true[:min_len]
    
    print(f"\n📊 评估数据规模: {len(y_true)}")
    print(f"   异常比例: {np.sum(y_true)/len(y_true):.2%}")
    
    # 6. 计算 AUC
    auc = roc_auc_score(y_true, y_pred_raw)
    print(f"\n🏆 ROC-AUC: {auc:.4f}")
    
    # 7. 计算 Point Adjustment F1
    # 这是一个耗时搜索，我们简化一下：先找最佳阈值，再应用 PA
    print("🔍 正在搜索最佳阈值 (Best F1)...")
    best_f1, best_thresh = get_best_f1(y_true, y_pred_raw)
    print(f"   Raw Best F1: {best_f1:.4f} (Threshold: {best_thresh:.6f})")
    
    print("🔧 应用 Point Adjustment (PA)...")
    y_pred_pa = point_adjustment(y_pred_raw, y_true, best_thresh)
    f1_pa = f1_score(y_true, y_pred_pa)
    
    print("\n" + "="*40)
    print(f"🌟 最终结果 (Final Result)")
    print("="*40)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"AUC    : {auc:.4f}")
    print(f"F1     : {best_f1:.4f} (Without PA)")
    print(f"F1-PA  : {f1_pa:.4f}  (With PA)")
    print("="*40)

    if auc > 0.8:
        print("✅ 恭喜！模型效果达到了预期水平！")
    else:
        print("⚠️ 效果一般，可能需要检查: 1. 窗口大小 2. 归一化方式 3. 训练Epoch数")

if __name__ == "__main__":
    main()