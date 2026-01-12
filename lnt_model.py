import torch
import torch.nn as nn


# === 这是一个标准的 LNT 核心模块 ===

class LNT_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        """
        :param input_dim: 输入特征数 (你的数据是 36)
        :param hidden_dim: 隐藏层大小 (比如 64)
        :param z_dim: 最终输出的特征向量维度 (比如 16)
        """
        super(LNT_Encoder, self).__init__()

        # 使用 LSTM 或 GRU 来提取时序特征
        # batch_first=True 意味着输入是 [Batch, Seq, Feature]
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 一个全连接层，把 RNN 的结果压缩成 z向量
        self.projection = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x 的形状: [Batch, 100, 36]

        # run_out 形状: [Batch, 100, hidden_dim]
        # h_n 形状: [1, Batch, hidden_dim] (这是最后时刻的隐藏状态)
        rnn_out, h_n = self.rnn(x)

        # 我们取最后一个时间点的状态作为这段窗口的“代表”
        # h_n 压缩掉第0维 -> [Batch, hidden_dim]
        last_hidden = h_n.squeeze(0)

        # 投影到潜在空间 z
        z = self.projection(last_hidden)

        return z  # 形状: [Batch, z_dim]


# === 模拟 LNT 的变换网络 (Transformation Network) ===
# 论文里说要预测变换，这里简化演示
class LNT_Transformation_Head(nn.Module):
    def __init__(self, z_dim, output_classes):
        super(LNT_Transformation_Head, self).__init__()
        self.classifier = nn.Linear(z_dim, output_classes)

    def forward(self, z):
        return self.classifier(z)