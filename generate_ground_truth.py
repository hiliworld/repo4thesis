import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

# === 配置 ===
INTERPRET_DIR = "./data/ServerMachineDataset/interpretation_label"
OUTPUT_DIR = "fault_clusters_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fault_ground_truth.csv")

def generate_gt():
    print("🚀 开始构建基于 Ground Truth 的故障数据集...")
    
    files = sorted(glob.glob(os.path.join(INTERPRET_DIR, "*.txt")))
    if not files:
        print("❌ 未找到文件，请检查路径。")
        return

    all_events = []
    fault_signatures = []

    # 1. 遍历文件解析
    for f_path in files:
        fname = os.path.basename(f_path)
        # 对应的原始数据文件名 (去掉了 .txt) 通常 SMD 的数据文件没有 .txt 后缀
        # interpretation_label/machine-1-1.txt -> machine-1-1
        raw_fname = fname.replace(".txt", "")
        
        with open(f_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 解析格式: "15849-16368:1,9,10,12,13,14,15"
            try:
                time_part, dim_part = line.split(':')
                start_idx, end_idx = map(int, time_part.split('-'))
                
                # 处理维度
                dims = [int(x) for x in dim_part.split(',')]
                dims.sort()
                
                # 生成指纹 (例如 "1-9-10-12...")
                signature = "-".join(map(str, dims))
                
                all_events.append({
                    'Machine_File': raw_fname,
                    'Start_Idx': start_idx,
                    'End_Idx': end_idx,
                    'Signature': signature
                })
                fault_signatures.append(signature)
                
            except Exception as e:
                # print(f"⚠️ 解析错误 ({fname}): {line} -> {e}")
                continue

    # 2. 统计故障类型并分配 ID
    print(f"✅ 解析完成，共提取到 {len(all_events)} 个故障事件。")
    
    # 统计频率
    counts = Counter(fault_signatures)
    
    # 排序：按频率从高到低
    sorted_sigs = [sig for sig, count in counts.most_common()]
    
    # 建立映射: Signature -> ID
    # 频率最高的 ID=0，次高的 ID=1，以此类推
    sig_to_id = {sig: idx for idx, sig in enumerate(sorted_sigs)}
    
    print("\n📊 故障类型统计 (Top 10):")
    print(f"{'ID':<4} | {'Count':<6} | {'Fault Fingerprint (Dimensions)'}")
    print("-" * 60)
    for idx, sig in enumerate(sorted_sigs[:10]):
        count = counts[sig]
        # 如果指纹太长，截断一下显示
        display_sig = (sig[:50] + '...') if len(sig) > 50 else sig
        print(f"{idx:<4} | {count:<6} | {display_sig}")
        
    # 3. 生成 CSV
    # 将 Signature 转换为 Cluster_Type (即 Fault ID)
    csv_data = []
    for event in all_events:
        event['Cluster_Type'] = sig_to_id[event['Signature']]
        csv_data.append(event)
        
    df = pd.DataFrame(csv_data)
    
    # 确保列的顺序符合 n_shot_experiment.py 的要求
    # 需要: Machine_File, Start_Idx, End_Idx, Cluster_Type
    df = df[['Machine_File', 'Start_Idx', 'End_Idx', 'Cluster_Type', 'Signature']]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 已保存 Ground Truth 文件至: {OUTPUT_FILE}")
    print("👉 现在，Cluster_Type 列代表真实的根因类别 ID (0=最常见, 1=次常见...)")
    
    # 4. 推荐划分
    print("\n💡 实验划分建议 (Copy to n_shot_experiment.py):")
    n_types = len(sorted_sigs)
    
    # 简单的频率划分建议
    # Train: 拿前 50% 高频的
    # Test: 拿中间的一些 (既不是只有1个样本的噪音，也不是这种特别常见的)
    
    train_ids = list(range(0, min(5, n_types))) 
    
    # 找一些样本数在 20-100 之间的作为测试集，比较合适
    test_candidates = []
    for idx, sig in enumerate(sorted_sigs):
        cnt = counts[sig]
        if 10 < cnt < 200 and idx not in train_ids:
            test_candidates.append(idx)
            
    if not test_candidates:
        test_candidates = list(range(5, min(10, n_types)))
        
    print(f"TRAIN_CLUSTERS = {train_ids}")
    print(f"TEST_CLUSTERS  = {test_candidates[:2]}  <-- 选一个作为 Target 跑跑看")

if __name__ == "__main__":
    generate_gt()