import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patches import PatchEmbedder


class NodePrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)

    def forward(self, z_fused: torch.Tensor):
        """
        Args:
            z_fused: [B, N, D]
        Returns:
            z_node_proto: [B, N, D]
            node_assign: [B, N, K]
            node_delta: [B, N, D]
        """
        z_norm = F.normalize(z_fused, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.transpose(0, 1))
        node_assign = F.softmax(sim / self.tau, dim=-1)
        z_node_proto = torch.matmul(node_assign, self.prototypes)
        node_delta = z_node_proto - z_fused
        return z_node_proto, node_assign, node_delta


class PatchPrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)

    def forward(self, z_patch: torch.Tensor):
        """
        Args:
            z_patch: [B, M, D]
        Returns:
            z_patch_proto: [B, M, D]
            patch_assign: [B, M, K]
            patch_delta: [B, M, D]
        """
        z_norm = F.normalize(z_patch, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.transpose(0, 1))
        patch_assign = F.softmax(sim / self.tau, dim=-1)
        z_patch_proto = torch.matmul(patch_assign, self.prototypes)
        patch_delta = z_patch_proto - z_patch
        return z_patch_proto, patch_assign, patch_delta


class PrototypeFusionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        use_node_prototype: bool = True,
        use_patch_prototype: bool = True,
        num_node_prototypes: int = 8,
        num_patch_prototypes: int = 8,
        tau_node: float = 0.1,
        tau_patch: float = 0.1,
        lambda_node: float = 0.5,
        lambda_patch: float = 0.3,
        patch_size: int = 10,
    ):
        super().__init__()
        self.use_node_prototype = use_node_prototype
        self.use_patch_prototype = use_patch_prototype
        self.lambda_node = lambda_node
        self.lambda_patch = lambda_patch

        self.node_bank = (
            NodePrototypeBank(latent_dim, num_node_prototypes, tau=tau_node)
            if use_node_prototype
            else None
        )
        self.patch_embedder = (
            PatchEmbedder(input_dim=input_dim, latent_dim=latent_dim, patch_size=patch_size)
            if use_patch_prototype
            else None
        )
        self.patch_bank = (
            PatchPrototypeBank(latent_dim, num_patch_prototypes, tau=tau_patch)
            if use_patch_prototype
            else None
        )

    def forward(self, x: torch.Tensor, z_fused: torch.Tensor):
        """
        Args:
            x: [B, T, N]
            z_fused: [B, N, D]
        Returns:
            dict with z_corrected and prototype intermediates
        """
        batch_size, num_nodes, latent_dim = z_fused.shape
        z_corrected = z_fused

        node_assign = None
        node_delta = torch.zeros_like(z_fused)
        z_node_proto = None

        patch_assign = None
        z_patch = None
        z_patch_proto = None
        patch_delta = None
        patch_global_delta = torch.zeros_like(z_fused)

        if self.use_node_prototype and self.node_bank is not None:
            z_node_proto, node_assign, node_delta = self.node_bank(z_fused)
            z_corrected = z_corrected + self.lambda_node * node_delta

        if self.use_patch_prototype and self.patch_embedder is not None and self.patch_bank is not None:
            z_patch, _ = self.patch_embedder(x)
            z_patch_proto, patch_assign, patch_delta = self.patch_bank(z_patch)
            patch_global = patch_delta.mean(dim=1)
            patch_global_delta = patch_global.unsqueeze(1).expand(batch_size, num_nodes, latent_dim)
            z_corrected = z_corrected + self.lambda_patch * patch_global_delta

        return {
            "z_corrected": z_corrected,
            "node_assign": node_assign,
            "patch_assign": patch_assign,
            "node_proto": z_node_proto,
            "patch_proto": z_patch_proto,
            "node_delta": node_delta,
            "patch_delta": patch_delta,
            "patch_global_delta": patch_global_delta,
            "z_patch": z_patch,
        }
