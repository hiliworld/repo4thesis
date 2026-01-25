import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === 组件 1: 位置编码 (保持不变) ===
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# === 组件 2: Metric Encoder (架构升级: GAT -> GRU) ===
# 【核心逻辑】使用 RNN 类结构来捕获时序的"突变"，而不是用 AvgPool 抹平它
class MetricGRUEncoder(nn.Module):
    def __init__(self, input_dim=333, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        # 1. 特征投影: 把 333 维的原始指标先压缩一下，方便 GRU 吃
        # 这一步相当于提取 spatial features
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 时序建模: GRU
        # batch_first=True -> Input: [Batch, SeqLen, Dim]
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. 输出投影
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # x: [Batch, Window, MetricDim] = [B, 20, 333]
        
        # 1. 预处理
        if mask is not None:
            # 如果有 mask，把无效维度的值清零 (虽然 dataset 里应该处理过了)
            x = x * mask.unsqueeze(1)
            
        # 2. 空间特征提取
        x_emb = self.feature_proj(x) # [B, 20, 128]
        
        # 3. 时序演化 (捕捉突变)
        # out: [B, 20, Hidden], hn: [Layers, B, Hidden]
        out, _ = self.gru(x_emb)
        
        # 4. 取最后一个时间步 (Last Step)
        # 代表了"读完这段波形后的最终状态"
        last_step_feat = out[:, -1, :] # [B, 64]
        
        # 5. 最终映射
        return self.norm(self.fc(last_step_feat))

# === 组件 3: Log Encoder (保持不变) ===
class LogAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=64, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            print("🧠 [Model] Loading Pretrained Semantic Vectors...")
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = True # 允许微调
            
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.count_proj = nn.Linear(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEmbedding(hidden_dim)
        
        # Log 依然用 Transformer，因为它更擅长处理语义组合
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask, log_count=None): 
        emb = self.embedding(x)
        h = self.proj(emb)
        h = h + self.pos_encoder(h)
        
        padding_mask = (mask == 0)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        
        # Pooling: 简单的加权平均
        mask_expanded = mask.unsqueeze(-1)
        sum_out = (h * mask_expanded).sum(dim=1)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        semantic_feat = sum_out / mask_sum 
        
        final_feat = self.fc(semantic_feat)
        
        if log_count is not None:
            count_feat = self.count_proj(log_count)
            count_feat = F.relu(count_feat)
            final_feat = final_feat + count_feat
            
        return self.norm(final_feat)

# === 组件 4: UAC 主模型 (适配新 Encoder) ===
class UACModel(nn.Module):
    def __init__(self, metric_dim=333, log_vocab_size=1000, embed_dim=64, log_weights=None):
        super().__init__()
        
        # 【修改】使用 GRU Encoder
        self.metric_encoder = MetricGRUEncoder(input_dim=metric_dim, hidden_dim=embed_dim)
        self.log_encoder = LogAttentionEncoder(vocab_size=log_vocab_size, hidden_dim=embed_dim, pretrained_weights=log_weights)
        
        self.mix_norm = nn.LayerNorm(embed_dim)
        
        self.metric_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.log_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, metric_seq, metric_mask, log_seq, log_mask, log_count=None, mixup_alpha=None):
        z_m = self.metric_encoder(metric_seq, metric_mask) # [B, 64]
        z_l = self.log_encoder(log_seq, log_mask, log_count) # [B, 64]
        
        batch_size = z_m.size(0)
        aux_info = {'mixup_active': False}

        if self.training and mixup_alpha is not None and mixup_alpha > 0 and batch_size > 1:
            beta_dist = torch.distributions.Beta(
                torch.tensor([mixup_alpha], device=z_m.device), 
                torch.tensor([mixup_alpha], device=z_m.device)
            )
            lam = beta_dist.sample().item() 
            lam = max(min(lam, 0.99), 0.01)
            
            index = torch.randperm(batch_size, device=z_m.device)
            
            z_m_mixed = lam * z_m + (1 - lam) * z_m[index]
            z_l_mixed = lam * z_l + (1 - lam) * z_l[index]
            
            z_m = self.mix_norm(z_m_mixed)
            z_l = self.mix_norm(z_l_mixed)
            
            aux_info = {
                'mixup_active': True,
                'lam': lam,
                'perm_index': index
            }

        p_m = self.metric_projector(z_m)
        p_l = self.log_projector(z_l)
        
        p_m = F.normalize(p_m, dim=1)
        p_l = F.normalize(p_l, dim=1)
        
        return p_m, p_l, aux_info