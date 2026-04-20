import torch
import torch.nn as nn
# 导入自定义模块
from src.models.encoders import LNT_Conv_Encoder
from src.models.layers import SpatialAttentionLayer 

class MyFinalModel(nn.Module):
    def __init__(self, config):
        super(MyFinalModel, self).__init__()
        
        # === 1. 基础配置 ===
        self.num_nodes = config['dataset']['input_dim'] # 节点数 (e.g., 38)
        self.window_size = config['dataset']['window_size']
        self.hidden_dim = config['model'].get('hidden_dim', 64)
        encoder_cfg = config.get('model', {}).get('encoder', {})
        
        # === 2. 核心组件初始化 ===
        
        # [组件 A] LNT Encoder
        # 根据你提供的代码，input_dim 应固定为 1 (单通道处理)，z_dim 对应 hidden_dim
        self.metric_encoder = LNT_Conv_Encoder(
            input_dim=1,           # 必须是 1，因为 forward 里强制 view(..., 1, seq_len)
            z_dim=self.hidden_dim, # 对应 LNT 代码里的 z_dim
            kernel_sizes=encoder_cfg.get('kernel_sizes', [3, 5, 9, 17]),
            head_channels=encoder_cfg.get('head_channels', 8),
            dropout=encoder_cfg.get('dropout', 0.1),
        )
        
        # [组件 B] GAT Layer
        # 根据你提供的代码，参数名为 input_dim, output_dim, num_heads
        self.gat_layer = SpatialAttentionLayer(
            input_dim=self.hidden_dim,   # 对应代码中的 input_dim
            output_dim=self.hidden_dim,  # 对应代码中的 output_dim
            num_heads=4,                 # 对应代码中的 num_heads
            dropout=0.2                  # 对应代码中的 dropout
        )
        
        # === 3. 任务头 ===
        
        # 预测头: [Hidden] -> [1]
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 重建头: [Hidden] -> [Window]
        self.recon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.window_size // 2),
            nn.ReLU(),
            nn.Linear(self.window_size // 2, self.window_size)
        )
        
        self.log_encoder = None

    def forward(self, x):
        """
        x: [Batch, Window, Nodes]
        """
        batch_size, window_size, num_nodes = x.shape
        
        # 1. LNT 提取特征
        # 输入: [Batch, Window, Nodes]
        # 输出: [Batch, Nodes, Hidden] (对应 LNT 代码最后的 z_nodes)
        z_local = self.metric_encoder(x)
        
        # 2. GAT 特征融合
        # 输入: [Batch, Nodes, Hidden]
        # 输出: out, weights (根据你的 GAT 代码 return out, avg_attn_weights)
        # 我们只需要 out 作为 z_global
        z_global, _ = self.gat_layer(z_local)
        
        # 3. 残差连接 (Fusion)
        z_combined = z_local + z_global 
        
        # 4. 任务输出
        
        # A. 预测头
        # input: [Batch, Nodes, Hidden] -> output: [Batch, Nodes, 1] -> [Batch, Nodes]
        pred_next = self.pred_head(z_combined).squeeze(-1)
        
        # B. 重建头
        # input: [Batch, Nodes, Hidden] -> output: [Batch, Nodes, Window]
        # permute -> [Batch, Window, Nodes] 匹配原始输入形状
        recon_window = self.recon_head(z_combined)
        recon_window = recon_window.permute(0, 2, 1) 
        
        # 返回 z_combined 用于可能的对比学习 Loss
        return pred_next, recon_window, z_combined
