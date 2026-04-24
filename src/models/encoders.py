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
        基于 LNT 的多头多尺度时序编码器 (legacy single-slot encoder)
        :param input_dim: 输入通道数
        :param z_dim: 输出特征维度
        """
        super(LNT_Conv_Encoder, self).__init__()
        kernel_sizes = kernel_sizes or [3, 5, 9, 17]
        self.heads = nn.ModuleList(
            [
                MultiScaleTemporalHead(
                    in_channels=input_dim,
                    out_channels=head_channels,
                    kernel_size=k,
                    dropout=dropout,
                )
                for k in kernel_sizes
            ]
        )
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
        # x: [B, T, N]
        batch_size, seq_len, num_features = x.shape
        # [B, T, N] -> [B*N, 1, T]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * num_features, 1, seq_len)

        head_outs = [head(x_reshaped) for head in self.heads]
        multi_scale = torch.cat(head_outs, dim=1)
        conv_out = self.fusion(multi_scale)
        conv_out = self.pool(conv_out)

        if self.pool_bins == 1:
            z_flat = conv_out.squeeze(-1)
        else:
            z_flat = conv_out.reshape(batch_size * num_features, -1)
            z_flat = self.pool_proj(z_flat)

        # [B*N, D] -> [B, N, D]
        z_nodes = z_flat.view(batch_size, num_features, -1)
        z_nodes = self.norm(z_nodes)
        return z_nodes


class SlotAggregator(nn.Module):
    """Aggregate slot-level node representation [B, N, S, D] -> [B, N, D]."""

    def __init__(self, hidden_dim: int, num_slots: int, mode: str = "gated_sum"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.mode = (mode or "gated_sum").lower()
        if self.mode not in ("gated_sum", "flatten_linear"):
            self.mode = "gated_sum"

        self.slot_gate = None
        self.flatten_proj = None
        if self.mode == "gated_sum":
            self.slot_gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.flatten_proj = nn.Linear(self.hidden_dim * self.num_slots, self.hidden_dim)

    def forward(self, z_slots):
        """
        Args:
            z_slots: [B, N, S, D]
        Returns:
            z_agg: [B, N, D]
            slot_weights: [B, N, S, 1]
        """
        if self.mode == "flatten_linear":
            bsz, num_nodes, num_slots, hidden_dim = z_slots.shape
            z_flat = z_slots.reshape(bsz, num_nodes, num_slots * hidden_dim)
            z_agg = self.flatten_proj(z_flat)
            slot_weights = torch.full(
                (bsz, num_nodes, num_slots, 1),
                fill_value=1.0 / max(1, num_slots),
                device=z_slots.device,
                dtype=z_slots.dtype,
            )
            return z_agg, slot_weights

        gate_logits = self.slot_gate(z_slots)
        slot_weights = torch.softmax(gate_logits, dim=2)
        z_agg = torch.sum(slot_weights * z_slots, dim=2)
        return z_agg, slot_weights


class LNTMultiSlotEncoder(nn.Module):
    """LNT-v2: Multi-slot local temporal encoder.

    Input:
        x: [B, T, N]
    Output:
        z_local_slots: [B, N, S, D]
    """

    def __init__(
        self,
        input_dim=1,
        z_dim=64,
        kernel_sizes=None,
        head_channels=8,
        dropout=0.1,
        num_slots=2,
        use_slot_pos_embedding=True,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 9, 17]
        self.num_slots = max(1, int(num_slots))
        self.use_slot_pos_embedding = bool(use_slot_pos_embedding)

        self.heads = nn.ModuleList(
            [
                MultiScaleTemporalHead(
                    in_channels=input_dim,
                    out_channels=head_channels,
                    kernel_size=k,
                    dropout=dropout,
                )
                for k in kernel_sizes
            ]
        )
        fused_in = len(kernel_sizes) * head_channels
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_in, z_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.slot_pool = nn.AdaptiveAvgPool1d(self.num_slots)
        self.norm = nn.LayerNorm(z_dim)
        # slot_embedding: [1, 1, S, D]
        self.slot_embedding = nn.Parameter(torch.zeros(1, 1, self.num_slots, z_dim))
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, T, N]
        bsz, seq_len, num_nodes = x.shape
        # [B, T, N] -> [B*N, 1, T]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(bsz * num_nodes, 1, seq_len)

        head_outs = [head(x_reshaped) for head in self.heads]
        multi_scale = torch.cat(head_outs, dim=1)  # [B*N, H, T]
        conv_out = self.fusion(multi_scale)  # [B*N, D, T]
        pooled = self.slot_pool(conv_out)  # [B*N, D, S]

        # [B*N, D, S] -> [B, N, S, D]
        z_local_slots = pooled.permute(0, 2, 1).contiguous().view(bsz, num_nodes, self.num_slots, -1)
        if self.use_slot_pos_embedding:
            z_local_slots = z_local_slots + self.slot_embedding
        z_local_slots = self.norm(z_local_slots)
        return z_local_slots
