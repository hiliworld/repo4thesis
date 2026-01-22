import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

# 引入之前的组件
from dataset import UACDataset
from model import UACModel

# === 配置 ===
CONFIG = {
    'data_dir': 'data/final_dataset',
    'gt_dir': 'data/Aiops-Dataset/groundtruth', 
    'batch_size': 256,
    'metric_window': 20,
    'metric_dim': 333,
    'embed_dim': 64,  # V4 模型参数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'checkpoints/uac_model_best.pth',
    # 【关键】这里建议保持较小的窗口，依靠 PA 来解决 F1 问题
    'anomaly_window_seconds': 60 * 1  # 1分钟窗口
}

# === 核心算法: Point Adjustment (分数修正版) ===
def apply_point_adjustment(y_true, y_scores):
    """
    实现 Point Adjustment 策略：
    如果一个故障区间内有任意时刻被预测为异常（分数高），
    则整个区间都被视为检测成功（将区间内的分数全部提升为该区间的最大分数）。
    这允许我们使用标准 API 计算最佳阈值。
    """
    y_scores_adjusted = y_scores.copy()
    
    # 找到 y_true 中所有的连续异常片段
    # 使用 diff 找到 0->1 和 1->0 的变化点
    changes = np.diff(np.r_[0, y_true, 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # 遍历每个故障片段
    for start, end in zip(starts, ends):
        # 找到该片段内模型预测的最大分数
        # 注意: end 是开区间，所以取切片 [start:end]
        segment_scores = y_scores[start:end]
        if len(segment_scores) == 0: continue
            
        max_score = np.max(segment_scores)
        
        # 将整个片段的分数修正为最大分数
        # 逻辑：只要这一段里有一个点爆了，就认为模型对这一整段都有信心
        y_scores_adjusted[start:end] = max_score
        
    return y_scores_adjusted

# === 1. 构建故障索引库 ===
def build_ground_truth_index(gt_dir):
    print(f"📚 [Ground Truth] 正在加载故障标签...")
    gt_files = glob.glob(os.path.join(gt_dir, "groundtruth-*.csv"))
    
    fault_index = {}
    total_faults = 0
    
    for f in gt_files:
        try:
            df = pd.read_csv(f)
            df = df.sort_values('timestamp')
            for _, row in df.iterrows():
                service = str(row['cmdb_id'])
                ts = int(row['timestamp'])
                if service not in fault_index:
                    fault_index[service] = []
                fault_index[service].append(ts)
                total_faults += 1
        except Exception as e:
            print(f"⚠️ 读取 GT 文件失败 {f}: {e}")
            
    print(f"✅ 故障库构建完成: 覆盖 {len(fault_index)} 个服务，共 {total_faults} 个故障记录。")
    return fault_index

# === 2. 生成标签 ===
def generate_labels(timestamps, service_name, fault_index):
    labels = np.zeros(len(timestamps))
    core_name = service_name.split('.')[0]
    if core_name in fault_index:
        fault_times = fault_index[core_name]
        for f_ts in fault_times:
            start_time = f_ts
            end_time = f_ts + CONFIG['anomaly_window_seconds']
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            labels[mask] = 1.0
    return labels

# === 3. 主评估流程 ===
def evaluate():
    # A. 准备模型
    print(f"🧠 [Model] Loading Pretrained Semantic Vectors...")
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0]
    else:
        vocab_size = 3000 
        weights = None
        
    model = UACModel(metric_dim=CONFIG['metric_dim'], 
                     log_vocab_size=vocab_size, 
                     embed_dim=CONFIG['embed_dim'],
                     log_weights=weights)

    try:
        print(f"📂 正在读取: {CONFIG['model_path']}")
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型权重加载成功！")
    except Exception as e:
        print(f"❌ 模型加载崩溃: {e}")
        return

    model.to(CONFIG['device'])
    model.eval()
    
    # B. 加载 Ground Truth
    fault_index = build_ground_truth_index(CONFIG['gt_dir'])
    
    # C. 遍历文件
    print("🚀 开始逐文件评估...")
    all_scores = []
    all_labels = []
    test_files = glob.glob(os.path.join(CONFIG['data_dir'], "*_test.pt"))
    
    if not test_files:
         print(f"❌ 未找到任何测试文件 (*_test.pt)")
         return

    for f_path in tqdm(test_files):
        try:
            content = torch.load(f_path, weights_only=False)
            metrics = content['metrics']
            logs = content['logs']
            timestamps = content['timestamps']
            mask = content['metric_mask']
            
            if len(metrics) < CONFIG['metric_window']: continue
            
            service_name = os.path.basename(f_path).replace("_test.pt", "")
            y_true = generate_labels(timestamps, service_name, fault_index)
            
            start_idx = CONFIG['metric_window']
            valid_len = len(metrics) - start_idx
            if valid_len <= 0: continue
            
            chunk_size = 256
            file_scores = []
            
            for i in range(0, valid_len, chunk_size):
                batch_m = []
                batch_l = []
                batch_l_mask = []
                real_batch_size = min(chunk_size, valid_len - i)
                
                for j in range(real_batch_size):
                    idx = start_idx + i + j
                    m_win = metrics[idx-CONFIG['metric_window']:idx]
                    batch_m.append(m_win)
                    
                    l_raw = logs[idx]
                    l_ts = torch.zeros(50, dtype=torch.long)
                    l_mk = torch.zeros(50, dtype=torch.float)
                    if len(l_raw) > 0:
                        slen = min(50, len(l_raw))
                        if isinstance(l_raw, np.ndarray):
                             l_ts[:slen] = torch.from_numpy(l_raw[:slen]).long()
                        else:
                             l_ts[:slen] = torch.tensor(l_raw[:slen]).long()
                        l_mk[:slen] = 1.0
                    batch_l.append(l_ts)
                    batch_l_mask.append(l_mk)
                
                if isinstance(batch_m[0], np.ndarray):
                     b_m = torch.from_numpy(np.stack(batch_m)).float().to(CONFIG['device'])
                else:
                     b_m = torch.stack(batch_m).float().to(CONFIG['device'])
                
                b_l = torch.stack(batch_l).to(CONFIG['device'])
                b_lm = torch.stack(batch_l_mask).to(CONFIG['device'])
                
                if isinstance(mask, np.ndarray):
                    mask_tensor = torch.from_numpy(mask).float()
                else:
                    mask_tensor = mask.float()
                b_mm = mask_tensor.unsqueeze(0).repeat(real_batch_size, 1).to(CONFIG['device'])
                
                with torch.no_grad():
                    p_m, p_l = model(b_m, b_mm, b_l, b_lm)
                    scores = 1.0 - (p_m * p_l).sum(dim=1)
                    file_scores.extend(scores.cpu().numpy())
            
            valid_labels = y_true[start_idx:]
            all_scores.extend(file_scores)
            all_labels.extend(valid_labels)
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    # === 4. 计算指标 ===
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"\n" + "="*50)
    print(f"📊 评估报告 | 样本总数: {len(all_labels)} | 窗口: {CONFIG['anomaly_window_seconds']}s")
    print(f"   - 异常 (P): {int(all_labels.sum())}")
    print(f"   - 正常 (N): {len(all_labels) - int(all_labels.sum())}")
    print("="*50)

    if all_labels.sum() == 0:
        print("❌ 测试集中无异常标签，无法计算指标。")
        return

    # --- Standard Metrics (Point-wise) ---
    print("\n🔍 [1. Standard Metrics (Point-wise)] - 严苛模式")
    try:
        auc = roc_auc_score(all_labels, all_scores)
        prec, rec, thresh = precision_recall_curve(all_labels, all_scores)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1)
        
        print(f"   🌟 AUC:       {auc:.4f}")
        print(f"   🏆 Best F1:   {f1[best_idx]:.4f}")
        print(f"      Precision: {prec[best_idx]:.4f}")
        print(f"      Recall:    {rec[best_idx]:.4f}")
        print(f"      Threshold: {thresh[best_idx] if best_idx < len(thresh) else thresh[-1]:.4f}")
    except:
        print("   ⚠️ 计算失败")

    # --- Point Adjustment Metrics (Event-wise) ---
    print("\n🚀 [2. Point Adjustment Metrics (Event-wise)] - 运维实战模式")
    print("   -> 正在应用 PA 策略修正分数...")
    
    # 核心步骤：修正分数
    all_scores_pa = apply_point_adjustment(all_labels, all_scores)
    
    try:
        # 使用修正后的分数计算 AUC 和 F1
        auc_pa = roc_auc_score(all_labels, all_scores_pa)
        prec_pa, rec_pa, thresh_pa = precision_recall_curve(all_labels, all_scores_pa)
        f1_pa = 2 * (prec_pa * rec_pa) / (prec_pa + rec_pa + 1e-8)
        best_idx_pa = np.argmax(f1_pa)
        
        print(f"   🌟 PA-AUC:    {auc_pa:.4f} (通常会极其接近 1.0)")
        print(f"   🏆 PA-F1:     {f1_pa[best_idx_pa]:.4f} <--- 关注这个!")
        print(f"      Precision: {prec_pa[best_idx_pa]:.4f}")
        print(f"      Recall:    {rec_pa[best_idx_pa]:.4f}")
        print(f"      Threshold: {thresh_pa[best_idx_pa] if best_idx_pa < len(thresh_pa) else thresh_pa[-1]:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ PA 计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate()