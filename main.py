import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# === 引入我们重构后的模块 ===
from src.data.loader import get_dataloaders
from src.models.anomaly_model import MyFinalModel
from src.utils.loss import ContrastiveLoss
from src.utils.metrics import get_best_f1

# === 全局设置 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ==========================================
# 🏋️ 训练流程 (Train)
# ==========================================
def train(args):
    config = load_config(args.config)
    print(f"🔥 Mode: TRAIN | Device: {DEVICE}")
    print(f"📜 Config: {args.config}")

    # 1. 准备数据
    train_loader, _, input_dim = get_dataloaders(args.config)
    config['dataset']['input_dim'] = input_dim # 动态更新维度
    
    # 2. 初始化模型
    model = MyFinalModel(config).to(DEVICE)
    
    # 3. 优化器与损失
    optimizer = optim.Adam(model.parameters(), lr=float(config['train']['lr']))
    criterion_mse = nn.MSELoss()
    criterion_cl = ContrastiveLoss(config['train']['batch_size'], device=DEVICE)
    
    # 4. 训练循环
    epochs = config['train']['epochs']
    patience = config['train']['patience']
    best_loss = float('inf')
    patience_counter = 0
    save_path = "best_model.pth" # 统一保存为这个名字

    print("\n🚀 Start Training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        start = time.time()
        
        for batch in train_loader:
            x = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward
            pred, recon, _ = model(x)
            
            # Loss Calculation
            # 任务A: 预测 (Target: x[:,-1,:])
            l_pred = criterion_mse(pred, x[:, -1, :])
            # 任务B: 重建 (Target: x)
            l_recon = criterion_mse(recon, x)
            
            # 任务C: 对比 (Augmentation & Encoder)
            # 简单的数据增强：加噪
            noise = torch.randn_like(x) * 0.01
            z1 = model.metric_encoder(x)
            z2 = model.metric_encoder(x + noise)
            l_cl = criterion_cl(z1.view(x.size(0), -1), z2.view(x.size(0), -1))
            
            # Total Loss
            loss = l_pred + l_recon + 0.1 * l_cl
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        cost = time.time() - start
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {cost:.1f}s")
        
        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved Best Model ({avg_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early Stopping Triggered.")
                break
                
    print(f"✅ Training Complete. Model saved to {save_path}")

# ==========================================
# 🧪 评估流程 (Evaluate)
# ==========================================
def evaluate(args):
    config = load_config(args.config)
    print(f"🔥 Mode: TEST | Device: {DEVICE}")
    
    # 1. 加载数据
    _, test_loader, input_dim = get_dataloaders(args.config)
    config['dataset']['input_dim'] = input_dim
    
    # 2. 加载模型
    model = MyFinalModel(config).to(DEVICE)
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} not found. Run train first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. 推理 (Inference)
    scores = []
    print("🚀 Running Inference...")
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(DEVICE)
            pred, recon, _ = model(x)
            
            # Score = Prediction Error + Reconstruction Error
            l_pred = torch.mean((pred - x[:, -1, :]) ** 2, dim=1)
            l_recon = torch.mean((recon - x) ** 2, dim=(1, 2))
            score = l_pred + l_recon
            scores.append(score.cpu().numpy())
            
    scores = np.concatenate(scores)
    
    # 4. 加载标签 (Ground Truth)
    # 这一步稍微复杂，需要去 dataset 目录找 label
    labels = load_labels(config)
    
    if labels is None:
        print("⚠️ No labels found. Skipping evaluation metrics.")
        return
        
    # 对齐长度
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]
   
    
    # 5. 计算指标
    print("📊 Calculating Metrics...")
    metrics = get_best_f1(labels, scores)
    
    print("\n" + "="*40)
    print(f"🌟 FINAL RESULTS ({config['dataset']['name']})")
    print("="*40)
    print(f"AUC       : {metrics['auc']:.4f}")
    print(f"Best F1   : {metrics['best_f1']:.4f}")
    print(f"PA F1     : {metrics['f1_pa']:.4f}")
    print("="*40)

def load_labels(config):
    """辅助函数：加载 SMD 标签"""
    test_path = config['dataset']['test_file']
    label_path = test_path.replace("test", "test_label")
    pattern = config['dataset']['format']['pattern']
    files = sorted(glob.glob(os.path.join(label_path, pattern)))
    
    label_list = []
    window = config['dataset']['window_size']
    
    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            raw = df.values.flatten()
            if len(raw) > window:
                label_list.append(raw[window:])
        except: pass
        
    if not label_list: return None
    return np.concatenate(label_list)

# ==========================================
# 🎯 主入口
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIOps Anomaly Detection Framework")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Run mode')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        evaluate(args)