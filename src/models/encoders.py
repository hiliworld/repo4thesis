import torch
import torch.nn as nn

class MultiScaleTemporalHead(nn.Module):
    """单个多尺度卷积分支。"""
    def __init__(self, in_channels=1, out_channels=8, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        mid_channels = max(out_channels // 2, 4)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class LNT_Conv_Encoder(nn.Module):
    def __init__(self, input_dim=1, z_dim=16, kernel_sizes=None, head_channels=8, dropout=0.1, pool_bins=1):
        """
        基于 LNT 的多头多尺度时序编码器
        :param input_dim: 输入通道数 (通常是 1，因为我们独立处理每个特征)
        :param z_dim: 输出特征维度
        """
        super(LNT_Conv_Encoder, self).__init__()
        kernel_sizes = kernel_sizes or [3, 5, 9, 17]
        self.heads = nn.ModuleList([
            MultiScaleTemporalHead(
                in_channels=input_dim,
                out_channels=head_channels,
                kernel_size=k,
                dropout=dropout,
            ) for k in kernel_sizes
        ])
        fused_in = len(kernel_sizes) * head_channels
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_in, z_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.pool_bins = int(pool_bins)
        if self.pool_bins not in (1, 2):
            self.pool_bins = 1
        self.pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.pool_proj = None
        if self.pool_bins == 2:
            self.pool_proj = nn.Linear(z_dim * 2, z_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x):
        # x: [Batch, Window, Features] -> [64, 100, 36]
        batch_size, seq_len, num_features = x.shape
        
        # 1. 维度变换：为了独立处理每个特征，我们需要把 Features 移到 Batch 维度
        # 目标格式: [Batch * Features, 1, Window]
        # 变换: [64, 100, 36] -> [64, 36, 100] -> [64*36, 1, 100]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * num_features, 1, seq_len)
        
        # 2. 多头多尺度卷积提取特征
        # out: [64*36, n_heads*head_channels, window]
        head_outs = [head(x_reshaped) for head in self.heads]
        multi_scale = torch.cat(head_outs, dim=1)
        conv_out = self.fusion(multi_scale)
        conv_out = self.pool(conv_out)

        # 3. 还原维度
        if self.pool_bins == 1:
            z_flat = conv_out.squeeze(-1)
        else:
            z_flat = conv_out.reshape(batch_size * num_features, -1)
            z_flat = self.pool_proj(z_flat)
        
        # view: [64, 36, z_dim] -> 变回 [Batch, Nodes, Features] 给 GAT 用
        z_nodes = z_flat.view(batch_size, num_features, -1)
        z_nodes = self.norm(z_nodes)
        
        return z_nodes
