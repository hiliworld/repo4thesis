import torch
import torch.nn as nn

class LNT_Conv_Encoder(nn.Module):
    def __init__(self, input_dim=1, z_dim=16):
        """
        基于 LNT 论文的卷积编码器
        :param input_dim: 输入通道数 (通常是 1，因为我们独立处理每个特征)
        :param z_dim: 输出特征维度
        """
        super(LNT_Conv_Encoder, self).__init__()
        
        # 定义卷积层序列
        # 结构参考 LNT network.py: Conv1d -> ReLU -> Conv1d ...
        # 我们调整了 stride 和 filter 以适应 window_size=100
        self.conv_net = nn.Sequential(
            # Layer 1: [Batch, 1, 100] -> [Batch, 4, 50] (Stride=2)
            nn.Conv1d(in_channels=input_dim, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 2: [Batch, 4, 50] -> [Batch, 8, 25] (Stride=2)
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: [Batch, 8, 25] -> [Batch, 16, 13] (Stride=2)
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 4: [Batch, 16, 13] -> [Batch, z_dim, 7] (Stride=2)
            nn.Conv1d(in_channels=16, out_channels=z_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 自适应池化：强制把时间维度变成 1
            # [Batch, z_dim, 7] -> [Batch, z_dim, 1]
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [Batch, Window, Features] -> [64, 100, 36]
        batch_size, seq_len, num_features = x.shape
        
        # 1. 维度变换：为了独立处理每个特征，我们需要把 Features 移到 Batch 维度
        # 目标格式: [Batch * Features, 1, Window]
        # 变换: [64, 100, 36] -> [64, 36, 100] -> [64*36, 1, 100]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * num_features, 1, seq_len)
        
        # 2. 卷积提取特征
        # out: [64*36, z_dim, 1]
        conv_out = self.conv_net(x_reshaped)
        
        # 3. 还原维度
        # squeeze: [64*36, z_dim]
        z_flat = conv_out.squeeze(-1)
        
        # view: [64, 36, z_dim] -> 变回 [Batch, Nodes, Features] 给 GAT 用
        z_nodes = z_flat.view(batch_size, num_features, -1)
        
        return z_nodes