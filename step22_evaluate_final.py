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
    'gt_dir': 'data/Aiops-Dataset/groundtruth', # Ground Truth 路径
    'batch_size': 256,
    'metric_window': 20,
    'metric_dim': 333,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'checkpoints/uac_model_best.pth',
    'anomaly_window_seconds': 60 * 5 # 故障发生后 5 分钟内都算异常
}

# === 1. 构建故障索引库 ===
def build_ground_truth_index(gt_dir):
    print(f"📚 [Ground Truth] 正在加载故障标签...")
    gt_files = glob.glob(os.path.join(gt_dir, "groundtruth-*.csv"))
    
    # 结构: fault_index['service_name'] = [timestamp1, timestamp2, ...]
    fault_index = {}
    total_faults = 0
    
    for f in gt_files:
        try:
            df = pd.read_csv(f)
            # 确保按时间排序
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
    """
    输入: 该 .pt 文件的时间戳序列, 服务名
    输出: 0/1 标签序列 (y_true)
    """
    labels = np.zeros(len(timestamps))
    
    # 提取核心服务名 (模糊匹配)
    # 例如: "frontend-0.source..." -> "frontend-0"
    core_name = service_name.split('.')[0]
    
    if core_name in fault_index:
        fault_times = fault_index[core_name]
        
        for f_ts in fault_times:
            # 找到 f_ts 在 timestamps 中的位置
            # 标记 [f_ts, f_ts + 5min] 范围内为 1
            start_time = f_ts
            end_time = f_ts + CONFIG['anomaly_window_seconds']
            
            # 向量化标记
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            labels[mask] = 1.0
            
    return labels

# === 3. 主评估流程 ===
def evaluate():
    # A. 准备模型
    
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0]
    else:
        vocab_size = 1000
        weights = None
    model = UACModel(metric_dim=CONFIG['metric_dim'], log_vocab_size=vocab_size, log_weights=weights)
    # === 原来的代码可能长这样 ===
    # model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    
    # === ✅ 请替换为以下 兼容性更强 的代码 ===
    try:
        print(f"📂 正在读取: {CONFIG['model_path']}")
        checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'], weights_only=False)
        
        # 判断: 如果 checkpoint 是一个字典且包含 'model' 键，说明是 Upgrade 2 格式
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("   -> 检测到 V2.0 格式 (包含 Loss 权重)，正在提取模型参数...")
            model.load_state_dict(checkpoint['model'])
        else:
            # 否则假设是旧版格式，直接加载
            print("   -> 检测到 V1.0 格式 (仅模型参数)...")
            model.load_state_dict(checkpoint)
            
        print("✅ 模型权重加载成功！")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {CONFIG['model_path']}")
        print("   -> 你运行完 train.py 了吗？有没有生成 uac_model_best.pth？")
        return
    except Exception as e:
        # 打印详细报错信息，而不是笼统的提示
        print(f"❌ 模型加载崩溃: {e}")
        import traceback
        traceback.print_exc()
        return
    except:
        print("❌ 模型加载失败，请检查路径")
        return
    model.to(CONFIG['device'])
    model.eval()
    
    # B. 加载 Ground Truth
    fault_index = build_ground_truth_index(CONFIG['gt_dir'])
    
    # C. 遍历测试集
    test_ds = UACDataset(CONFIG['data_dir'], mode='test', metric_window=CONFIG['metric_window'])
    # 为了能对应文件名，我们需要 Dataset 暴露文件名信息
    # 这里我们简化处理：UACDataset 是把所有文件拼在一起的
    # 但 global_samples 里没有存文件名。
    # 💡 补救：为了能做 Label，我们需要单独加载每个文件进行评估，而不能用合并的 DataLoader。
    
    print("🚀 开始逐文件评估 (File-by-File Evaluation)...")
    
    all_scores = []
    all_labels = []
    
    # 手动扫描文件列表
    test_files = glob.glob(os.path.join(CONFIG['data_dir'], "*_test.pt"))
    
    for f_path in tqdm(test_files):
        try:
            # 1. 加载单个文件数据
            content = torch.load(f_path, weights_only=False)
            metrics = content['metrics']
            logs = content['logs']
            timestamps = content['timestamps']
            mask = content['metric_mask']
            
            if len(metrics) < CONFIG['metric_window']: continue
            
            # 2. 生成 y_true
            service_name = os.path.basename(f_path).replace("_test.pt", "")
            y_true = generate_labels(timestamps, service_name, fault_index)
            
            # 3. 运行模型 (Batch推理)
            # 构造临时 Batch
            # 这里简单起见，我们按顺序做滑窗
            # 为了速度，我们手动拼 Batch
            
            # 提取特征
            m_seqs = []
            l_seqs = []
            l_masks = []
            
            # 仅评估 valid 部分
            start_idx = CONFIG['metric_window']
            valid_len = len(metrics) - start_idx
            if valid_len <= 0: continue
            
            # 构造输入 Tensor
            # 注意：这里如果数据量大，显存可能会爆。我们分块处理。
            chunk_size = 256
            file_scores = []
            
            for i in range(0, valid_len, chunk_size):
                batch_m = []
                batch_l = []
                batch_l_mask = []
                
                real_batch_size = min(chunk_size, valid_len - i)
                
                for j in range(real_batch_size):
                    idx = start_idx + i + j
                    # Metric Window
                    m_win = metrics[idx-CONFIG['metric_window']:idx]
                    batch_m.append(m_win)
                    
                    # Log Sequence
                    l_raw = logs[idx]
                    l_ts = torch.zeros(50, dtype=torch.long)
                    l_mk = torch.zeros(50, dtype=torch.float)
                    if len(l_raw) > 0:
                        slen = min(50, len(l_raw))
                        l_ts[:slen] = torch.from_numpy(l_raw[:slen])
                        l_mk[:slen] = 1.0
                    batch_l.append(l_ts)
                    batch_l_mask.append(l_mk)
                
                # 转 Tensor
                b_m = torch.stack(batch_m).to(CONFIG['device'])
                b_l = torch.stack(batch_l).to(CONFIG['device'])
                b_lm = torch.stack(batch_l_mask).to(CONFIG['device'])
                # mask 是一样的
                b_mm = mask.unsqueeze(0).repeat(real_batch_size, 1).to(CONFIG['device'])
                
                # 推理
                with torch.no_grad():
                    p_m, p_l = model(b_m, b_mm, b_l, b_lm)
                    scores = 1.0 - (p_m * p_l).sum(dim=1)
                    file_scores.extend(scores.cpu().numpy())
            
            # 4. 收集结果
            # y_true 也要切掉前 window 大小
            valid_labels = y_true[start_idx:]
            
            all_scores.extend(file_scores)
            all_labels.extend(valid_labels)
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            
    # === 4. 计算最终指标 ===
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"\n📊 最终评估报告 (Total Samples: {len(all_labels)})")
    print(f"   - 异常样本数 (Positive): {int(all_labels.sum())}")
    print(f"   - 正常样本数 (Negative): {len(all_labels) - int(all_labels.sum())}")
    
    if all_labels.sum() == 0:
        print("❌ 警告：在测试集中没有匹配到任何故障标签！(Precision/Recall 无法计算)")
        return

    # 计算 AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"🌟 AUC-ROC Score: {auc:.4f}")
    
    # 寻找最佳 F1 的阈值
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    print(f"🏆 Best F1 Score: {f1_scores[best_idx]:.4f}")
    print(f"   - Threshold: {thresholds[best_idx]:.4f}")
    print(f"   - Precision: {precisions[best_idx]:.4f}")
    print(f"   - Recall:    {recalls[best_idx]:.4f}")

if __name__ == "__main__":
    evaluate()