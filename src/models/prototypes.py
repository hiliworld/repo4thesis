import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patches import PatchEmbedder


class PrototypeMatcher(nn.Module):
    """Shared cosine matcher for prototype assignment and reconstruction."""

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, z: torch.Tensor, prototypes: torch.Tensor):
        """
        Args:
            z: [B, Q, D]
            prototypes: [K, D]
        Returns:
            z_proto: [B, Q, D]
            assign: [B, Q, K]
            delta: [B, Q, D]
        """
        z_norm = F.normalize(z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.transpose(0, 1))
        assign = F.softmax(sim / self.tau, dim=-1)
        z_proto = torch.matmul(assign, prototypes)
        delta = z_proto - z
        return z_proto, assign, delta


class PrototypeCorrection(nn.Module):
    """Fuse node/patch prototype corrections into z_corrected."""

    def __init__(
        self,
        latent_dim: int,
        lambda_node: float = 0.5,
        lambda_patch: float = 0.3,
        use_node_correction: bool = True,
        use_patch_correction: bool = True,
        fusion_mode: str = "residual_add",
    ):
        super().__init__()
        self.lambda_node = lambda_node
        self.lambda_patch = lambda_patch
        self.use_node_correction = use_node_correction
        self.use_patch_correction = use_patch_correction
        self.fusion_mode = fusion_mode

        self.linear_fuse = None
        if fusion_mode == "linear_fuse":
            self.linear_fuse = nn.Linear(latent_dim * 3, latent_dim)

    def forward(self, z_fused: torch.Tensor, node_delta: torch.Tensor, patch_global_delta: torch.Tensor, z_node_proto: torch.Tensor):
        if self.fusion_mode == "linear_fuse":
            node_guided = z_fused + self.lambda_node * node_delta if self.use_node_correction else z_fused
            patch_guided = patch_global_delta if self.use_patch_correction else torch.zeros_like(z_fused)
            z_cat = torch.cat([z_fused, node_guided, patch_guided], dim=-1)
            return self.linear_fuse(z_cat)

        z_corrected = z_fused
        if self.use_node_correction:
            z_corrected = z_corrected + self.lambda_node * node_delta
        if self.use_patch_correction:
            z_corrected = z_corrected + self.lambda_patch * patch_global_delta
        return z_corrected


class NodePrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)
        self.matcher = PrototypeMatcher(tau=tau)

    def forward(self, z_fused: torch.Tensor):
        return self.matcher(z_fused, self.prototypes)


class PatchPrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)
        self.matcher = PrototypeMatcher(tau=tau)

    def forward(self, z_patch: torch.Tensor):
        return self.matcher(z_patch, self.prototypes)


class PrototypeFusionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        use_node_prototype: bool = True,
        use_patch_prototype: bool = True,
        use_node_correction: bool = True,
        use_patch_correction: bool = True,
        num_node_prototypes: int = 8,
        num_patch_prototypes: int = 8,
        tau_node: float = 0.1,
        tau_patch: float = 0.1,
        lambda_node: float = 0.5,
        lambda_patch: float = 0.3,
        patch_size: int = 10,
        fusion_mode: str = "residual_add",
    ):
        super().__init__()
        self.use_node_prototype = use_node_prototype
        self.use_patch_prototype = use_patch_prototype

        self.node_bank = NodePrototypeBank(latent_dim, num_node_prototypes, tau=tau_node) if use_node_prototype else None
        self.patch_embedder = PatchEmbedder(input_dim=input_dim, latent_dim=latent_dim, patch_size=patch_size) if use_patch_prototype else None
        self.patch_bank = PatchPrototypeBank(latent_dim, num_patch_prototypes, tau=tau_patch) if use_patch_prototype else None

        self.correction = PrototypeCorrection(
            latent_dim=latent_dim,
            lambda_node=lambda_node,
            lambda_patch=lambda_patch,
            use_node_correction=use_node_correction,
            use_patch_correction=use_patch_correction,
            fusion_mode=fusion_mode,
        )

    def forward(self, x: torch.Tensor, z_fused: torch.Tensor):
        batch_size, num_nodes, latent_dim = z_fused.shape

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

        if self.use_patch_prototype and self.patch_embedder is not None and self.patch_bank is not None:
            z_patch, _ = self.patch_embedder(x)
            z_patch_proto, patch_assign, patch_delta = self.patch_bank(z_patch)
            patch_global = patch_delta.mean(dim=1)
            patch_global_delta = patch_global.unsqueeze(1).expand(batch_size, num_nodes, latent_dim)

        z_corrected = self.correction(
            z_fused=z_fused,
            node_delta=node_delta,
            patch_global_delta=patch_global_delta,
            z_node_proto=z_node_proto,
        )

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
