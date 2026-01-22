import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === 组件 1: 位置编码 (Positional Embedding) ===
# 作用: 给 Log 序列打上时间戳，让模型理解 "先后顺序"
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

# === 组件 2: 交叉注意力 (预埋) ===
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value, key_padding_mask=None):
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        return self.norm(query + attn_out)

# === 核心 1: Metric Encoder (TCN + Dynamic Graph) ===
# 改进: 结合了 TCN 提取时序，Self-Attention 提取指标间依赖
class MetricGATEncoder(nn.Module):
    def __init__(self, input_dim=333, hidden_dim=64, window_size=20):
        super().__init__()
        
        # A. 时序特征提取 (Temporal Convolution)
        # 先把每个指标看作独立的时间序列，压缩时间维度 T=20 -> T=1
        # 这就像给每个指标算了一个"加权特征值"
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, 
                      kernel_size=3, padding=1, groups=input_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # [B, 333, 20] -> [B, 333, 1]
        )
        
        # B. 维度投影
        # 将每个指标的标量特征映射到高维空间，方便算 Attention
        # 333 个节点，每个节点现在有 64 维的特征
        self.feature_proj = nn.Linear(1, hidden_dim) 
        
        # C. 动态图注意力 (Dynamic Graph Learning)
        # 这里虽然用的是 MultiheadAttention，但我们输入的 Sequence Length 是 333 (指标数量)
        # 这意味着我们在计算 "指标 i" 和 "指标 j" 的相关性 -> 这就是 GAT 的本质！
        self.graph_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # D. 读出层 (Readout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        # x: [Batch, Window, 333] -> 转置为 [Batch, 333, Window]
        x = x.permute(0, 2, 1) 
        
        # 1. 时序压缩: [Batch, 333, 1]
        # 每个指标变成了一个特征点
        t_feat = self.temporal_conv(x)
        
        # 2. 特征投影: [Batch, 333, 1] -> [Batch, 333, 64]
        # 现在我们有 333 个节点，每个节点由 64 维向量表示
        nodes = self.feature_proj(t_feat)
        
        # 3. 图注意力交互 (Graph Interaction)
        # 这一步让模型自动学习：CPU Load (Node i) 是否应该关注 Disk IO (Node j)
        # 加上 residual connection 防止梯度消失
        nodes_updated, _ = self.graph_attn(nodes, nodes, nodes)
        nodes = nodes + nodes_updated
        
        # 4. Masked Pooling (图读出)
        # 将 333 个节点的特征聚合成 1 个图特征向量
        if mask is not None:
            # mask: [Batch, 333] -> [Batch, 333, 1]
            mask = mask.unsqueeze(-1)
            nodes = nodes * mask # 屏蔽掉 Padding 的指标
            valid_count = mask.sum(dim=1).clamp(min=1.0)
            out = nodes.sum(dim=1) / valid_count # 平均池化
        else:
            out = nodes.mean(dim=1)
            
        return self.fc(out)

# === 核心 2: Log Encoder (Transformer + Positional) ===
# 改进: 引入 Positional Embedding，彻底利用参考代码 log_model_v3.py 的优势
class LogAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=64, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if pretrained_weights is not None:
            print("🧠 [Model] Loading Pretrained Semantic Vectors...")
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = True 
            
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
        # 【关键升级】位置编码
        self.pos_encoder = PositionalEmbedding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        # x: [Batch, 50]
        emb = self.embedding(x) # [B, 50, 384]
        h = self.proj(emb)      # [B, 50, 64]
        
        # 注入位置信息：让模型知道日志发生的先后顺序
        h = h + self.pos_encoder(h)
        
        # Transformer 处理
        padding_mask = (mask == 0)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        
        # 聚合
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sum_out = (h * mask.unsqueeze(-1)).sum(dim=1)
        avg_out = sum_out / mask_sum
        
        return self.fc(avg_out)

# === 核心 3: UAC 主模型 ===
class UACModel(nn.Module):
    def __init__(self, metric_dim=333, log_vocab_size=1000, log_weights=None):
        super().__init__()
        
        self.metric_encoder = MetricGATEncoder(input_dim=metric_dim)
        self.log_encoder = LogAttentionEncoder(vocab_size=log_vocab_size, pretrained_weights=log_weights)
        
        # 预留 Upgrade 1 接口
        self.cross_attn = CrossAttention(embed_dim=64)
        
        self.metric_projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.log_projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, metric_seq, metric_mask, log_seq, log_mask):
        z_m = self.metric_encoder(metric_seq, metric_mask)
        z_l = self.log_encoder(log_seq, log_mask)
        
        p_m = self.metric_projector(z_m)
        p_l = self.log_projector(z_l)
        
        p_m = F.normalize(p_m, dim=1)
        p_l = F.normalize(p_l, dim=1)
        
        return p_m, p_l