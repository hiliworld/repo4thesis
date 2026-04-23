import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSegmentEncoder(nn.Module):
    """Learnable anomaly-space encoder.

    Window-level:
        Inputs:
            z_local_slots: [B, N, S, D]
            z_global_slots: [B, N, S, D]
            z_corrected: [B, N, D]
            score_stats: [B, K] or None
        Output:
            window_repr: [B, E]

    Segment-level:
        Input:
            window_repr_seq: [L, E]
        Output:
            segment_repr: [E]
    """

    def __init__(self, hidden_dim: int, repr_dim: int = 64, score_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.repr_dim = int(repr_dim)
        self.score_dim = int(score_dim)

        in_dim = self.hidden_dim * 3 + self.score_dim
        self.window_projector = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.repr_dim),
        )
        self.window_norm = nn.LayerNorm(self.repr_dim)

        self.segment_attn = nn.Sequential(
            nn.Linear(self.repr_dim, self.repr_dim),
            nn.Tanh(),
            nn.Linear(self.repr_dim, 1),
        )

    def encode_window(self, z_local_slots, z_global_slots, z_corrected, score_stats=None):
        # z_local_slots: [B, N, S, D]
        # z_global_slots: [B, N, S, D]
        # z_corrected: [B, N, D]
        # score_stats: [B, K]
        local_summary = z_local_slots.mean(dim=(1, 2))
        global_summary = z_global_slots.mean(dim=(1, 2))
        corrected_summary = z_corrected.mean(dim=1)

        if score_stats is None:
            score_stats = torch.zeros(
                (z_corrected.size(0), self.score_dim),
                dtype=z_corrected.dtype,
                device=z_corrected.device,
            )
        elif score_stats.size(-1) < self.score_dim:
            pad = torch.zeros(
                (score_stats.size(0), self.score_dim - score_stats.size(-1)),
                dtype=score_stats.dtype,
                device=score_stats.device,
            )
            score_stats = torch.cat([score_stats, pad], dim=-1)
        elif score_stats.size(-1) > self.score_dim:
            score_stats = score_stats[:, : self.score_dim]

        feat = torch.cat([local_summary, global_summary, corrected_summary, score_stats], dim=-1)
        window_repr = self.window_projector(feat)
        window_repr = self.window_norm(window_repr)
        window_repr = F.normalize(window_repr, p=2, dim=-1)
        return window_repr

    def encode_segment(self, window_repr_seq, mode="attentive"):
        # window_repr_seq: [L, E]
        if window_repr_seq.ndim != 2:
            raise ValueError(f"window_repr_seq must be 2D [L,E], got shape={tuple(window_repr_seq.shape)}")

        if window_repr_seq.size(0) == 0:
            return torch.zeros((self.repr_dim,), dtype=window_repr_seq.dtype, device=window_repr_seq.device)

        mode = (mode or "attentive").lower()
        if mode == "mean":
            seg = window_repr_seq.mean(dim=0)
            return F.normalize(seg, p=2, dim=-1)

        attn_logits = self.segment_attn(window_repr_seq)  # [L, 1]
        attn = torch.softmax(attn_logits, dim=0)
        seg = torch.sum(attn * window_repr_seq, dim=0)
        return F.normalize(seg, p=2, dim=-1)
