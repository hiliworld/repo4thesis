import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    """Extract fixed-size temporal patches from x:[B,T,N] and project to latent dim D."""

    def __init__(self, input_dim: int, latent_dim: int, patch_size: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.projector = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, N]
        Returns:
            z_patch: [B, M, D]
            x_patch: [B, M, P, N]
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D tensor [B,T,N], got shape={tuple(x.shape)}")

        batch_size, time_len, num_nodes = x.shape
        if num_nodes != self.input_dim:
            raise ValueError(
                f"PatchEmbedder input_dim mismatch: expected {self.input_dim}, got {num_nodes}"
            )

        num_patches = time_len // self.patch_size
        if num_patches < 1:
            raise ValueError(
                f"window size ({time_len}) is smaller than patch_size ({self.patch_size})"
            )

        effective_len = num_patches * self.patch_size
        x_trimmed = x[:, :effective_len, :]
        x_patch = x_trimmed.view(batch_size, num_patches, self.patch_size, num_nodes)

        patch_pooled = x_patch.mean(dim=2)
        z_patch = self.projector(patch_pooled)
        return z_patch, x_patch
