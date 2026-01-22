import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import math

# 引入组件
from dataset import UACDataset
from model import UACModel

# === 配置 (V4.0 Projector 适配版) ===
CONFIG = {
    'data_dir': 'data/final_dataset',
    'batch_size': 512,      # 3090Ti 显存巨大，直接拉到 512
    'epochs': 200,          # 跑满 100 轮
    'patience': 10,         # 耐心给足一点 (5 -> 10)，对比学习收敛慢
    'lr': 3e-4,             # 【关键】稍微调大 LR 到 3e-4
    'min_lr': 1e-6,         # 余弦退火的最低学习率
    'metric_dim': 333,
    'metric_window': 20,
    'embed_dim': 64,        # 【新增】和 model.py 里的 hidden_dim 保持一致
    'log_vocab_size': 0,    # 自动读取
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# === 核心组件 1: 可学习温度的 InfoNCE Loss ===
class LearnableInfoNCELoss(nn.Module):
    def __init__(self, init_temp=0.07, max_temp=100.0):
        super().__init__()
        # log(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        self.max_temp = max_temp
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_a, features_b):
        # 1. 限制温度范围
        # clamp logit_scale 防止 exp 溢出
        self.logit_scale.data.clamp_(min=0, max=4.6) # exp(4.6) ≈ 100
        
        logit_scale = self.logit_scale.exp()
        
        # 2. 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.T) * logit_scale
        
        # 3. 生成标签
        labels = torch.arange(logits.shape[0]).to(logits.device)
        
        # 4. 双向 Loss
        loss_m2l = self.criterion(logits, labels)
        loss_l2m = self.criterion(logits.T, labels)
        
        return loss_m2l, loss_l2m

# === 核心组件 2: 动态不确定性加权 Loss ===
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

# === 核心组件 3: 对抗训练 ===
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='embedding'):
        # 针对 Embedding 层进行扰动
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
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

# === 主训练循环 ===
def train():
    print(f"🚀 [Training V4.0] 设备: {CONFIG['device']} | Projector: ON | LR: {CONFIG['lr']}")
    
    # 1. 准备数据
    train_ds = UACDataset(CONFIG['data_dir'], mode='train', metric_window=CONFIG['metric_window'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True)
    
    print(f"   样本数: {len(train_ds)} | Steps: {len(train_loader)}")

    # 2. 准备模型
    # 尝试加载语义向量
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0]
        print(f"✅ 加载语义向量，词表大小: {vocab_size}")
    else:
        # 如果没找到，给一个足够大的默认值 (3000 应该够了)
        vocab_size = 3000 
        weights = None
        print("⚠️ 未找到预训练向量，使用随机初始化 (Vocab=3000)")

    # 【适配点】传入 embed_dim
    model = UACModel(metric_dim=CONFIG['metric_dim'], 
                     log_vocab_size=vocab_size, 
                     embed_dim=CONFIG['embed_dim'],
                     log_weights=weights)
                     
    if torch.cuda.device_count() > 1:
        print(f"🔥 使用 {torch.cuda.device_count()} 张 GPU 进行并行训练")
        model = nn.DataParallel(model)
        
    model = model.to(CONFIG['device'])
    
    # 3. 准备 Loss 和 优化器
    criterion_base = LearnableInfoNCELoss(init_temp=0.07).to(CONFIG['device'])
    criterion_dynamic = DynamicWeightedLoss(num_losses=2).to(CONFIG['device'])
    
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3},
        {'params': criterion_base.parameters(), 'lr': 1e-3}
    ], lr=CONFIG['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr']
    )
    
    # FGM 需要解包 DataParallel 才能找到 parameter
    fgm_model = model.module if isinstance(model, nn.DataParallel) else model
    fgm = FGM(fgm_model)
    
    best_loss = float('inf')
    patience_counter = 0

    # 4. Epoch 循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_m2l = 0
        total_l2m = 0
        
        # 打印状态
        if isinstance(criterion_base, nn.DataParallel):
            temp_mod = criterion_base.module
        else:
            temp_mod = criterion_base
            
        with torch.no_grad():
            precisions = torch.exp(-criterion_dynamic.params)
            w_m = precisions[0].item()
            w_l = precisions[1].item()
            current_temp = 1.0 / temp_mod.logit_scale.exp().item()
            current_lr = optimizer.param_groups[0]['lr']
            
        print(f"\n📊 [Epoch {epoch+1}] Temp: {current_temp:.4f} | LR: {current_lr:.6f} | Weights -> Metric: {w_m:.2f} Log: {w_l:.2f}")

        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch in pbar:
            m_seq = batch['metric_seq'].to(CONFIG['device'])
            m_mask = batch['metric_mask'].to(CONFIG['device'])
            l_seq = batch['log_seq'].to(CONFIG['device'])
            l_mask = batch['log_mask'].to(CONFIG['device'])
            
            # --- A. 正常训练 ---
            optimizer.zero_grad()
            p_m, p_l = model(m_seq, m_mask, l_seq, l_mask)
            
            loss_m2l, loss_l2m = criterion_base(p_m, p_l)
            loss = criterion_dynamic([loss_m2l, loss_l2m])
            
            loss.backward()
            
            # --- B. 对抗训练 ---
            fgm.attack(epsilon=0.1) 
            p_m_adv, p_l_adv = model(m_seq, m_mask, l_seq, l_mask)
            loss_m2l_adv, loss_l2m_adv = criterion_base(p_m_adv, p_l_adv)
            loss_adv = criterion_dynamic([loss_m2l_adv, loss_l2m_adv])
            loss_adv.backward()
            fgm.restore()
            
            # 梯度裁剪 (Projector 容易梯度大，加个 Clip 很重要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_m2l += loss_m2l.item()
            total_l2m += loss_l2m.item()
            
            pbar.set_postfix({
                'L': f"{loss.item():.3f}", 
                'M2L': f"{loss_m2l.item():.3f}", 
                'Temp': f"{current_temp:.3f}"
            })
        
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        
        print(f"📉 Avg Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # 处理 DataParallel 保存问题
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            state = {
                'model': model_state,
                'loss_params': criterion_dynamic.state_dict(),
                'temp_param': temp_mod.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(CONFIG['save_dir'], 'uac_model_best.pth'))
            print("💾 Best Model Saved.")
        else:
            patience_counter += 1
            print(f"⚠️ Loss 未下降 ({patience_counter}/{CONFIG['patience']})")
            
            if patience_counter >= CONFIG['patience']:
                print(f"🛑 触发早停! 训练结束。")
                break

if __name__ == "__main__":
    train()