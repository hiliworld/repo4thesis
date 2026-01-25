import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
import time

from dataset import UACDataset
from model import UACModel

# === 配置 (V6.4 Final Combine: Fixed Temp + Masking) ===
CONFIG = {
    'data_dir': 'data/final_dataset',
    'batch_size': 512,      
    'epochs': 200,           
    'patience': 10,         
    'lr': 3e-4,             
    'min_lr': 1e-6,         
    'metric_dim': 333,
    'metric_window': 20,
    'embed_dim': 64,        
    'log_vocab_size': 0,    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints',
    'mixup_alpha': 0.0      # One-Class 模式下保持关闭
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# === 组件 1: 固定温度 + 智能去重 Loss (最强组合) ===
class FixedMaskedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07): 
        super().__init__()
        self.temperature = temperature # 固定温度，拒绝偷懒
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_a, features_b, log_seqs=None):
        """
        features_a: [B, Dim] (Metric)
        features_b: [B, Dim] (Log)
        log_seqs: [B, Len] (Log 原始内容，用于去重)
        """
        batch_size = features_a.shape[0]
        device = features_a.device
        
        # 1. 计算相似度矩阵
        # logits: [B, B]
        logits = torch.matmul(features_a, features_b.T) / self.temperature
        
        # 2. 构建掩码 (Mask)
        # 基础掩码：自己和自己不是负样本
        final_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 【核心逻辑回归】内容重复掩码
        # 如果 Log_i 和 Log_j 内容一样，它们就是“友军”，不能当负样本推开
        if log_seqs is not None:
            # diff: [B, B, L]
            diff = log_seqs.unsqueeze(1) - log_seqs.unsqueeze(0)
            is_duplicate = (diff == 0).all(dim=-1) # [B, B]
            final_mask = final_mask | is_duplicate
            
        # 3. 处理 Logits
        # 将 Mask 掉的位置设为 -inf (Softmax 后概率为 0)
        # 这样 Loss 就只会计算“真正的负样本”
        # 注意：对角线(正样本)需要保留吗？
        # CrossEntropyLoss 需要 logits 和 target。
        # 这里我们手动实现 InfoNCE 的分母部分处理会更灵活，
        # 但为了配合 CrossEntropyLoss，我们可以把 masked 的负样本 logits 设为极其小
        
        # Trick: 我们希望正样本(对角线)保持原值，负样本(非对角线但重复的)设为 -inf
        # CrossEntropyLoss 的 target 是 label=0,1,2... (即对角线)
        # 所以我们只需要把 "非对角线 且 重复" 的位置设为 -inf
        
        # 重新构建 mask: 只屏蔽 "非对角线 且 重复" 的
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        duplicate_mask = torch.zeros_like(diag_mask)
        if log_seqs is not None:
            diff = log_seqs.unsqueeze(1) - log_seqs.unsqueeze(0)
            duplicate_mask = (diff == 0).all(dim=-1)
        
        # 真正要屏蔽的：是重复的，但不是自己 (即同类互斥的情况)
        mask_to_ignore = duplicate_mask & (~diag_mask)
        
        # 应用屏蔽
        logits = logits.masked_fill(mask_to_ignore, -1e9)
        
        # 生成标签 (对角线是正样本)
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        
        # 计算 Loss
        # Metric -> Log
        loss_a = self.criterion(logits, labels)
        # Log -> Metric (对称)
        loss_b = self.criterion(logits.T, labels)
        
        return loss_a, loss_b

# === 组件 2: 动态权重 (不变) ===
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

# === 组件 3: FGM (不变) ===
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
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

# === 主训练循环 ===
def train():
    print(f"🚀 [Training] V6.4 Final Combo | Fixed Temp=0.07 | With Masking")
    start_time = time.time()

    # 1. 权重
    emb_path = "data/processed_logs/log_semantic_embeddings.pth"
    if os.path.exists(emb_path):
        weights = torch.load(emb_path)
        vocab_size = weights.shape[0] 
        print(f"✅ 加载语义向量: {vocab_size}")
    else:
        vocab_size = 3000 
        print("⚠️ 使用随机初始化向量")
        weights = None

    # 2. 数据
    train_ds = UACDataset(CONFIG['data_dir'], mode='train', 
                          metric_window=CONFIG['metric_window'],
                          vocab_size=vocab_size)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True)
    
    # 3. 模型
    model = UACModel(metric_dim=CONFIG['metric_dim'], 
                     log_vocab_size=vocab_size,
                     embed_dim=CONFIG['embed_dim'],
                     log_weights=weights)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    model = model.to(CONFIG['device'])
    
    # 4. 工具
    # 【核心】使用 固定温度 + Masking 的 Loss
    criterion_base = FixedMaskedInfoNCELoss(temperature=0.07).to(CONFIG['device'])
    criterion_dynamic = DynamicWeightedLoss(num_losses=2).to(CONFIG['device'])
    
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_dynamic.parameters(), 'lr': 1e-3}
    ], lr=CONFIG['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    fgm = FGM(model.module if isinstance(model, nn.DataParallel) else model)
    
    best_loss = float('inf')
    patience_counter = 0

    print("-" * 80)
    print(f"{'Epoch':^6} | {'Lr':^10} | {'Temp':^7} | {'Avg Loss':^10} | {'Metric Std':^10} | {'Log Std':^10} | {'Status':^10}")
    print("-" * 80)

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        m_std_val, l_std_val = 1.0, 1.0 

        for batch_idx, batch in enumerate(pbar):
            m_seq = batch['metric_seq'].to(CONFIG['device'])
            m_mask = batch['metric_mask'].to(CONFIG['device'])
            l_seq = batch['log_seq'].to(CONFIG['device'])
            l_mask = batch['log_mask'].to(CONFIG['device'])
            l_count = batch['log_count'].to(CONFIG['device']) if 'log_count' in batch else None

            optimizer.zero_grad()
            
            # Forward
            p_m, p_l, aux_info = model(m_seq, m_mask, l_seq, l_mask, log_count=l_count, mixup_alpha=None)
            
            # Diagnostic
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    m_std_val = F.normalize(p_m, dim=1).std(dim=0).mean().item()
                    l_std_val = F.normalize(p_l, dim=1).std(dim=0).mean().item()

            # Loss (必须传入 log_seqs 进行去重!)
            loss_m2l, loss_l2m = criterion_base(p_m, p_l, log_seqs=l_seq) 
            loss = criterion_dynamic([loss_m2l, loss_l2m])
            loss.backward()
            
            # FGM
            fgm.attack(epsilon=0.1) 
            p_m_adv, p_l_adv, _ = model(m_seq, m_mask, l_seq, l_mask, log_count=l_count, mixup_alpha=None)
            loss_m2l_adv, loss_l2m_adv = criterion_base(p_m_adv, p_l_adv, log_seqs=l_seq)
            loss_adv = criterion_dynamic([loss_m2l_adv, loss_l2m_adv])
            loss_adv.backward()
            fgm.restore()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'L': f"{loss.item():.3f}", 'MS': f"{m_std_val:.2f}", 'LS': f"{l_std_val:.2f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        status_icon = "🟢" if avg_loss < best_loss else "⚪"
        print(f"{epoch+1:^6} | {current_lr:.1e}  | {'0.0700':^7} | {avg_loss:.4f}     | {m_std_val:.4f}     | {l_std_val:.4f}     | {status_icon}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            state = {
                'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'loss_params': criterion_dynamic.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(CONFIG['save_dir'], 'uac_model_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"🛑 Early stopping.")
                break

    print(f"⏱️  Total Time: {(time.time()-start_time)/60:.1f} min")

if __name__ == "__main__":
    train()