import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import math

# 引入组件 (假设 dataset.py 和 model.py 都在同一目录下)
from dataset import UACDataset
from model import UACModel

# === 配置 (V3.2 暴力突破版) ===
CONFIG = {
    'data_dir': 'data/final_dataset',
    'batch_size': 512,      # 【关键】3090Ti 显存巨大，直接拉到 512！
    'epochs': 300,           # 跑满 50 轮
    'patience': 20,         # 耐心给足一点
    'lr': 1e-4,             # 【关键】初始学习率调小，防止震荡
    'min_lr': 1e-6,         # 【新增】余弦退火的最低学习率
    'metric_dim': 333,
    'metric_window': 20,
    'log_vocab_size': 0,    # 自动读取
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# === 核心组件 1: 可学习温度的 InfoNCE Loss (Upgrade!) ===
class LearnableInfoNCELoss(nn.Module):
    def __init__(self, init_temp=0.07, max_temp=100.0):
        super().__init__()
        # 使用 log 形式存储温度，保证数值稳定且恒为正
        # log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        self.max_temp = max_temp
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_a, features_b):
        # 1. 限制最大温度，防止梯度爆炸 (Optional but safe)
        # exp(logit_scale) 等价于 1/temperature
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=self.max_temp)
        
        # 2. 计算相似度矩阵 (CosSim * scale)
        # features 必须是归一化过的 [B, Dim]
        logits = torch.matmul(features_a, features_b.T) * logit_scale
        
        # 3. 生成标签 (对角线是正样本)
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        # 4. 双向计算 Loss
        loss_m2l = self.criterion(logits, labels)       # Metric找Log
        loss_l2m = self.criterion(logits.T, labels)     # Log找Metric
        
        return loss_m2l, loss_l2m

# === 核心组件 2: 动态不确定性加权 Loss (保持不变) ===
class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_losses=2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            log_var = self.params[i]
            loss_sum += 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
        return loss_sum

# === 核心组件 3: 对抗训练 (保持不变) ===
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# === 主训练循环 ===
def train():
    print(f"🚀 [Training V3.2] 设备: {CONFIG['device']} | 模式: Learnable Temp + Cosine Scheduler")
    
    # 1. 准备数据
    train_ds = UACDataset(CONFIG['data_dir'], mode='train', metric_window=CONFIG['metric_window'])
    # 注意：num_workers 设置为 4-8 可以加快数据加载，pin_memory=True 加速转 GPU
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True)
    
    # 2. 准备模型
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0]
        print(f"✅ 加载语义向量，词表大小: {vocab_size}")
    else:
        vocab_size = 1000
        weights = None
        print("⚠️ 未找到预训练向量，使用随机初始化")

    model = UACModel(metric_dim=CONFIG['metric_dim'], log_vocab_size=vocab_size, log_weights=weights)
    model = model.to(CONFIG['device'])
    
    # 3. 准备 Loss 和 优化器
    # 【改动】实例化可学习 Loss
    criterion_base = LearnableInfoNCELoss(init_temp=0.07).to(CONFIG['device'])
    criterion_dynamic = DynamicWeightedLoss(num_losses=2).to(CONFIG['device'])
    
    # 优化器需要同时管理：模型参数、DynamicLoss参数、InfoNCE温度参数
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3},
        {'params': criterion_base.parameters(), 'lr': 1e-3} # 温度的学习率可以给大一点
    ], lr=CONFIG['lr'])
    
    # 【改动】定义余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr']
    )
    
    fgm = FGM(model)
    best_loss = float('inf')
    patience_counter = 0

    # 4. Epoch 循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_m2l = 0
        total_l2m = 0
        
        # 打印当前状态 (权重 + 温度 + 学习率)
        with torch.no_grad():
            precisions = torch.exp(-criterion_dynamic.params)
            w_m = precisions[0].item()
            w_l = precisions[1].item()
            # 计算当前实际温度 = 1 / exp(logit_scale)
            current_temp = 1.0 / criterion_base.logit_scale.exp().item()
            current_lr = optimizer.param_groups[0]['lr']
            
        print(f"\n📊 [Epoch {epoch+1}] Temp: {current_temp:.4f} | LR: {current_lr:.6f} | Weights -> Metric: {w_m:.2f} Log: {w_l:.2f}")

        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch in pbar:
            m_seq = batch['metric_seq'].to(CONFIG['device'])
            m_mask = batch['metric_mask'].to(CONFIG['device'])
            l_seq = batch['log_seq'].to(CONFIG['device'])
            l_mask = batch['log_mask'].to(CONFIG['device'])
            
            # --- A. 正常训练步 ---
            optimizer.zero_grad()
            p_m, p_l = model(m_seq, m_mask, l_seq, l_mask)
            
            # 获取两个分量 Loss
            loss_m2l, loss_l2m = criterion_base(p_m, p_l)
            
            # 动态加权
            loss = criterion_dynamic([loss_m2l, loss_l2m])
            
            loss.backward()
            
            # --- B. 对抗训练步 (FGM) ---
            fgm.attack(epsilon=0.1) 
            p_m_adv, p_l_adv = model(m_seq, m_mask, l_seq, l_mask)
            loss_m2l_adv, loss_l2m_adv = criterion_base(p_m_adv, p_l_adv)
            loss_adv = criterion_dynamic([loss_m2l_adv, loss_l2m_adv])
            loss_adv.backward()
            fgm.restore()
            
            optimizer.step()
            
            # 记录数据
            total_loss += loss.item()
            total_m2l += loss_m2l.item()
            total_l2m += loss_l2m.item()
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}", 
                'M2L': f"{loss_m2l.item():.3f}", 
                'L2M': f"{loss_l2m.item():.3f}"
            })
        
        # 【改动】更新学习率
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_m2l = total_m2l / len(train_loader)
        avg_l2m = total_l2m / len(train_loader)
        
        print(f"📉 Avg Loss: {avg_loss:.4f} (M->L: {avg_m2l:.4f}, L->M: {avg_l2m:.4f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            state = {
                'model': model.state_dict(),
                'loss_params': criterion_dynamic.state_dict(),
                'temp_param': criterion_base.state_dict() # 保存温度参数
            }
            torch.save(state, os.path.join(CONFIG['save_dir'], 'uac_model_best.pth'))
            print("💾 Best Model Saved.")
        else:
            patience_counter += 1
            print(f"⚠️ Loss 未下降 ({patience_counter}/{CONFIG['patience']})")
            
            if patience_counter >= CONFIG['patience']:
                print(f"🛑 触发早停机制! 训练结束。最佳 Loss: {best_loss:.4f}")
                break

if __name__ == "__main__":
    train()