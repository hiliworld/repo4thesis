import torch
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === 配置 ===
CONFIG = {
    'template_file': 'data/processed_logs/templates.csv', 
    'output_path': 'data/processed_logs/log_semantic_embeddings.pth',
    
    # 【关键修改】指向你的本地文件夹名
    # 只要这个文件夹里有 config.json 和 model.safetensors 即可
    'model_name': './bert_model',  
    
    'vocab_size': 3000
}

def generate_embeddings():
    print(f"🚀 [Semantic] 启动！正在加载本地 BERT 模型: {CONFIG['model_name']}...")
    
    # SentenceTransformer 支持直接加载本地路径
    try:
        model = SentenceTransformer(CONFIG['model_name'])
    except Exception as e:
        print(f"❌ 加载本地模型失败: {e}")
        print("   -> 请确认 'bert_model' 文件夹就在当前目录下，且包含模型文件。")
        return
    
    print(f"📂 正在读取模板文件: {CONFIG['template_file']}")
    if not os.path.exists(CONFIG['template_file']):
        print(f"❌ 错误: 找不到文件 {CONFIG['template_file']}")
        return

    try:
        df = pd.read_csv(CONFIG['template_file'])
        print(f"   -> 成功读取 {len(df)} 条日志模板")
        
        if 'EventId' not in df.columns or 'Template' not in df.columns:
            print(f"❌ 列名不匹配！你的CSV包含: {df.columns.tolist()}")
            return
            
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    embed_dim = 384
    embedding_matrix = torch.zeros((CONFIG['vocab_size'], embed_dim))
    
    max_id = df['EventId'].max()
    if max_id >= CONFIG['vocab_size']:
        print(f"⚠️ 警告: 最大 EventID ({max_id}) 超过配置，自动调整 vocab_size")
        CONFIG['vocab_size'] = max_id + 100
        embedding_matrix = torch.zeros((CONFIG['vocab_size'], embed_dim))

    print("🔄 开始生成语义向量...")
    count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        event_id = int(row['EventId'])
        template_text = str(row['Template'])
        clean_text = template_text.replace('<*>', 'parameter').strip()
        
        with torch.no_grad():
            vector = model.encode(clean_text)
        
        if event_id < CONFIG['vocab_size']:
            embedding_matrix[event_id] = torch.tensor(vector)
            count += 1
            
    print(f"✅ 处理完成！生成 {count} 个向量。")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    torch.save(embedding_matrix, CONFIG['output_path'])
    print(f"💾 已保存至: {CONFIG['output_path']}")

if __name__ == "__main__":
    generate_embeddings()