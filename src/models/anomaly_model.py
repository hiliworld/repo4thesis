import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoders import LNT_Conv_Encoder

class MyFinalModel(nn.Module):
    def __init__(self, config):
        """
        通用模型：目前仅启用数值(Metric)模式
        """
        super(MyFinalModel, self).__init__()
        
        # 强制默认为 metric，或者从配置读
        self.modality = config['dataset'].get('modality', 'metric')
        self.hidden_dim = config['model']['hidden_dim']
        
        print(f"🤖 初始化模型模式: {self.modality.upper()}")

        # ==========================
        # 🏗️ 分支 A: 数值型 (SMD/SWaT)
        # ==========================
        if self.modality == 'metric':
            # 这里的 input_dim 会在 step8 中被动态覆盖为 37
            self.feature_dim = config['dataset'].get('input_dim', 38)
            self.window_size = config['dataset']['window_size']
            
            # 【修复点】正确实例化 LNT_Conv_Encoder
            # LNT_Conv_Encoder 只接受 input_dim (默认1) 和 z_dim
            # 注意：它的 input_dim 指的是卷积的通道数，而在 forward 里我们把它reshape成了 (B*N, 1, W)
            # 所以这里的 input_dim 应该是 1
            self.metric_encoder = LNT_Conv_Encoder(
                input_dim=1, 
                z_dim=config['model']['z_dim']
            )
            
            # 预测头 (Forecasting)
            # Encoder 输出 z 的维度是 [Batch, Nodes, z_dim]
            # 我们需要把 z_dim 映射回 1 (预测下一个值) 或者其他逻辑
            # 原来的代码可能是：self.pred_head = nn.Linear(self.hidden_dim, self.feature_dim)
            # 但现在 metric_encoder 输出的是 [B, N, z_dim]
            
            # 让我们看看 forward 怎么写的：
            # z = self.metric_encoder(x) -> [B, N, z_dim]
            
            # 现在的 pred_head 需要把 z_dim 变成 1 (预测该 Sensor 的下一个值)
            # 输入: [B, N, z_dim] -> Linear -> [B, N, 1] -> squeeze -> [B, N]
            self.pred_head = nn.Linear(config['model']['z_dim'], 1)
            
            # 重建头 (Reconstruction)
            # 输入: [B, N, z_dim] -> Linear -> [B, N, Window]
            self.recon_head = nn.Linear(config['model']['z_dim'], self.window_size)

        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def forward(self, x):
        if self.modality == 'metric':
            # --- 数值路径 ---
            # x: [Batch, Window, Features] -> [B, W, N]
            # 注意：PyTorch RNN 习惯 [B, W, N]，但 LNT Encoder 需要 [B, W, N]
            
            # 1. 编码
            # z: [Batch, Nodes, z_dim]
            z = self.metric_encoder(x) 
            
            # 2. 任务 A: 预测下一个点 (Forecasting)
            # [B, N, z_dim] -> [B, N, 1] -> [B, N]
            pred_next = self.pred_head(z).squeeze(-1)
            
            # 3. 任务 B: 重建整个窗口 (Reconstruction)
            # [B, N, z_dim] -> [B, N, W] -> Permute -> [B, W, N]
            recon_window = self.recon_head(z).permute(0, 2, 1)
            
            return pred_next, recon_window, z

        else:
            raise ValueError(f"Unknown modality: {self.modality}")