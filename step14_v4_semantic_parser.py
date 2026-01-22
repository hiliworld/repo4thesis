import os
import glob
import pandas as pd
import numpy as np
import torch
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# === 配置 ===
DATA_ROOT = "data/Aiops-Dataset/data"
OUTPUT_DIR = "data/processed_logs"
DRAIN_CONFIG = "drain3.ini"
# 由于网络问题 这里采用本地下载的使用方式
BERT_MODEL_NAME = './bert_model'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_and_embed_logs():
    print(f"🧠 [Log Parser SOTA版] 启动！将引入语义理解能力...")
    
    # 1. 初始化 Drain3
    config = TemplateMinerConfig()
    config.load(DRAIN_CONFIG)
    template_miner = TemplateMiner(persistence_handler=None, config=config)
    
    # 2. 扫描文件
    search_pattern = os.path.join(DATA_ROOT, "*", "log", "all", "*.csv")
    log_files = glob.glob(search_pattern)
    print(f"   -> 发现 {len(log_files)} 个日志文件")
    
    # 3. 第一遍扫描：建立模板库 (只解析，不保存)
    # 为了得到全局统一的 ID，必须先看一遍所有数据
    print("   -> Phase 1: 扫描全量日志构建模板库...")
    for file_path in tqdm(log_files, desc="Building Templates"):
        try:
            df = pd.read_csv(file_path)
            # 确保 value 是字符串
            contents = df['value'].dropna().astype(str).tolist()
            for line in contents:
                template_miner.add_log_message(line)
        except Exception as e:
            pass

    # 4. 生成语义向量 (Semantic Embedding)
    print("   -> Phase 2: 使用 BERT 生成模板向量 (这可能需要一点时间)...")
    # 加载 BERT
    bert_model = SentenceTransformer(BERT_MODEL_NAME)
    
    # 获取所有模板
    clusters = template_miner.drain.clusters
    vocab_size = len(clusters) + 1 # +1 是因为 ID 0 要留给 Padding/Unknown
    embedding_dim = 384 # MiniLM 的输出维度是 384
    
    # 初始化矩阵 [Vocab, Dim]
    # row 0 是全0向量 (Padding)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    templates_text = []
    cluster_ids = []
    
    for cluster in clusters:
        # Drain 的 ID 是从 1 开始的，直接用作索引
        idx = cluster.cluster_id
        template_str = cluster.get_template()
        
        templates_text.append(template_str)
        cluster_ids.append(idx)
    
    # 批量编码 (Batch Encoding) 比逐条快得多
    if templates_text:
        embeddings = bert_model.encode(templates_text, show_progress_bar=True)
        # 填入矩阵
        for i, idx in enumerate(cluster_ids):
            embedding_matrix[idx] = embeddings[i]
            
    # 保存语义矩阵
    emb_path = os.path.join(OUTPUT_DIR, "log_semantic_embeddings.pth")
    torch.save(torch.from_numpy(embedding_matrix), emb_path)
    print(f"✅ 语义矩阵已保存: {emb_path} | Shape: {embedding_matrix.shape}")

    # 5. 第二遍扫描：保存解析结果
    print("   -> Phase 3: 保存解析后的 Event ID...")
    log_buffer = {}
    
    for file_path in tqdm(log_files, desc="Saving IDs"):
        try:
            df = pd.read_csv(file_path)
            # 同样做筛选
            df = df[['timestamp', 'cmdb_id', 'value']].dropna()
            
            for _, row in df.iterrows():
                ts = row['timestamp']
                if ts > 1e11: ts = ts // 1000
                content = str(row['value']).strip()
                cmdb_id = row['cmdb_id']
                
                # 匹配模板 (match) 而不是添加 (add)
                cluster = template_miner.match(content)
                if cluster:
                    event_id = cluster.cluster_id
                else:
                    event_id = 0 # 没见过的或者是噪音
                
                if cmdb_id not in log_buffer:
                    log_buffer[cmdb_id] = []
                
                log_buffer[cmdb_id].append({
                    "timestamp": ts,
                    "event_id": event_id
                })
        except:
            pass
            
    # 保存 CSV
    for cmdb_id, events in tqdm(log_buffer.items(), desc="Writing CSV"):
        safe_name = cmdb_id.replace("/", "_").replace(":", "_")
        save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
        df_out = pd.DataFrame(events)
        df_out.sort_values('timestamp', inplace=True)
        df_out.to_csv(save_path, index=False)

    print("🎉 SOTA 级日志解析完成！")

if __name__ == "__main__":
    parse_and_embed_logs()