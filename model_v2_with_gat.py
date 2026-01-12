import torch
import torch.nn as nn
import torch.nn.functional as F


# === 1. 独立的 Encoder (保持不变) ===
class LNT_Independent_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(LNT_Independent_Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x: [Batch, 100, 36]
        batch_size, seq_len, num_features = x.shape
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * num_features, seq_len, 1)
        rnn_out, h_n = self.rnn(x_reshaped)
        last_hidden = h_n.squeeze(0)
        z_flat = self.projection(last_hidden)
        z_nodes = z_flat.view(batch_size, num_features, -1)
        return z_nodes


# === 2. 简单的 GAT 层 (保持不变) ===
class SimpleGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(SimpleGATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        batch_size, num_nodes, _ = h.shape
        wh = self.W(h)
        wh_repeated_in_chunks = wh.repeat_interleave(num_nodes, dim=1)
        wh_repeated_alternating = wh.repeat(1, num_nodes, 1)
        all_combinations = torch.cat([wh_repeated_in_chunks, wh_repeated_alternating], dim=2)
        e = self.leakyrelu(self.a(all_combinations))
        attention = F.softmax(e.view(batch_size, num_nodes, num_nodes), dim=2)
        h_prime = torch.bmm(attention, wh)
        return F.elu(h_prime), attention


# === 3. 【重点修改】最终模型 (增加重建头) ===
class MyFinalModel(nn.Module):
    def __init__(self, num_features=36, window_size=100):
        super(MyFinalModel, self).__init__()
        self.num_features = num_features
        self.window_size = window_size

        # A. 编码器
        self.lnt_encoder = LNT_Independent_Encoder(input_dim=1, hidden_dim=64, z_dim=16)

        # B. GAT 层
        self.gat_layer = SimpleGATLayer(in_features=16, out_features=16)

        # C. 预测头 (Forecasting Head): 预测下一时刻的值 (36个值)
        self.pred_head = nn.Linear(num_features * 16, num_features)

        # D. 【新增】重建头 (Reconstruction Head)
        # 目标：从 GAT 增强后的特征，还原回 [Batch, 100, 36]
        # 这里为了简化，我们先用一个简单的 MLP 把特征映射回 100 个点
        self.recon_decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, window_size)  # 输出 100 个点
        )

    def forward(self, x):
        # x: [Batch, 100, 36]
        batch_size, seq_len, num_features = x.shape

        # 1. 提取特征 -> [Batch, 36, 16]
        node_features = self.lnt_encoder(x)

        # 2. GAT 关联 -> [Batch, 36, 16]
        gat_features, attn_weights = self.gat_layer(node_features)

        # 3. 任务一：预测未来 (Forecasting)
        # 把所有节点展平 -> [Batch, 36*16] -> [Batch, 36]
        pred_next = self.pred_head(gat_features.view(batch_size, -1))

        # 4. 【新增】任务二：重建历史 (Reconstruction)
        # gat_features: [Batch, 36, 16]
        # 我们想对每个节点单独重建它的 100 个时间点
        # view -> [Batch * 36, 16]
        recon_flat = self.recon_decoder(gat_features.view(batch_size * num_features, -1))
        # recon_flat: [Batch * 36, 100]
        # 变回形状 -> [Batch, 36, 100] -> 转置 -> [Batch, 100, 36]
        recon_window = recon_flat.view(batch_size, num_features, seq_len).permute(0, 2, 1)

        return pred_next, recon_window, attn_weights