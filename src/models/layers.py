import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    """
    基于多头注意力机制的空间相关性建模层 (Adaptive Spatial Attention)
    论文中的角色：捕获不同传感器(Nodes)之间的动态关联
    """
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.2):
        super(SpatialAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # 确保输出维度能被头数整除
        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = output_dim // num_heads
        
        # 定义 Q, K, V 投影矩阵
        # 这里的 Linear 是作用在特征维度上的
        self.W_Q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_K = nn.Linear(input_dim, output_dim, bias=False)
        self.W_V = nn.Linear(input_dim, output_dim, bias=False)
        
        # 最终的输出融合层
        self.W_out = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子 (Scale Factor)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        输入 x: [Batch, Nodes, Features]
        例如: [64, 36, 16] -> 表示 64 个时刻，36 个机器指标，每个指标有 16 维的 LNT 特征
        """
        batch_size, num_nodes, _ = x.shape
        
        # 1. 线性投影 + 分头 (Linear Projection & Split Heads)
        # [B, N, D] -> [B, N, Heads, Head_Dim] -> [B, Heads, N, Head_Dim]
        Q = self.W_Q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # 核心：计算节点与节点之间的相似度
        # Q @ K.T: [B, Heads, N, Head_Dim] @ [B, Heads, Head_Dim, N] -> [B, Heads, N, N]
        # 这个 [N, N] 就是我们要学习的“邻接矩阵”！
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Softmax 归一化，得到概率分布
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout
        attn_weights = self.dropout(attn_weights)
        
        # 3. 聚合信息 (Aggregation)
        # Weights @ V: [B, Heads, N, N] @ [B, Heads, N, Head_Dim] -> [B, Heads, N, Head_Dim]
        out = torch.matmul(attn_weights, V)
        
        # 4. 拼接多头并输出 (Concat Heads)
        # [B, Heads, N, Head_Dim] -> [B, N, Heads * Head_Dim] -> [B, N, Output_Dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_dim)
        
        # 最后的线性变换 + 残差连接准备
        out = self.W_out(out)
        
        # 返回 output 和 attention_weights (便于可视化分析图结构)
        # 我们取所有头的平均 Attention 作为可视化的依据
        avg_attn_weights = attn_weights.mean(dim=1) # [Batch, N, N]
        
        return out, avg_attn_weights