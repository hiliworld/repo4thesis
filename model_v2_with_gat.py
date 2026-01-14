import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. 导入组件 ===
try:
    from lnt_encoder import LNT_Conv_Encoder
    from adaptive_gat import SpatialAttentionLayer  # <--- ✅ 新增：导入刚才写好的文件
except ImportError:
    print("❌ 错误：找不到 lnt_encoder.py 或 adaptive_gat.py")
    exit()

# (SimpleGATLayer 可以删掉了，或者留着做纪念，反正我们不用它了)

# === 2. 最终模型 (完全体) ===
class MyFinalModel(nn.Module):
    def __init__(self, num_features=36, window_size=100, hidden_dim=64, z_dim=16):
        """
        :param z_dim: 编码后的特征维度 (LNT 输出维度)
        """
        super(MyFinalModel, self).__init__()
        
        # A. LNT 编码器 (Step 7 已完成)
        # 作用：提取局部时序特征
        self.lnt_encoder = LNT_Conv_Encoder(input_dim=1, z_dim=z_dim)

        # B. 自适应空间注意力层 (Step 9 核心升级) <--- ✅ 修改点
        # 作用：自动学习传感器之间的关联图
        # 我们让 output_dim = z_dim，保持维度一致方便计算
        self.gat_layer = SpatialAttentionLayer(
            input_dim=z_dim, 
            output_dim=z_dim, 
            num_heads=4,    # 使用 4 个头，分别关注不同的关系模式
            dropout=0.2
        )

        # C. 预测头
        self.pred_head = nn.Linear(num_features * z_dim, num_features)

        # D. 重建头
        self.recon_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size)
        )

    def forward(self, x):
        # x: [Batch, 100, 36]
        batch_size, seq_len, num_features = x.shape

        # 1. LNT 编码
        # out: [Batch, 36, 16]
        node_features = self.lnt_encoder(x)

        # 2. 自适应 GAT 融合 <--- ✅ 修改点
        # out: [Batch, 36, 16] (特征融合后)
        # attn: [Batch, 36, 36] (学习到的关系图)
        gat_features, attn_weights = self.gat_layer(node_features)

        # 3. 预测 (使用融合了空间信息的特征)
        pred_next = self.pred_head(gat_features.view(batch_size, -1))

        # 4. 重建 (对每个特征独立重建)
        gat_flat = gat_features.view(batch_size * num_features, -1)
        recon_flat = self.recon_decoder(gat_flat)
        recon_window = recon_flat.view(batch_size, num_features, seq_len).permute(0, 2, 1)

        return pred_next, recon_window, attn_weights