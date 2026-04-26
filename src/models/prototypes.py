import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patches import PatchEmbedder


class PrototypeMatcher(nn.Module):
    """Shared cosine matcher for prototype assignment and reconstruction."""

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, z: torch.Tensor, prototypes: torch.Tensor, return_extra: bool = False):
        """
        Args:
            z: [B, Q, D]
            prototypes: [K, D]
        Returns:
            z_proto: [B, Q, D]
            assign: [B, Q, K]
            delta: [B, Q, D]
            extra(optional): dict
        """
        z_norm = F.normalize(z, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.transpose(0, 1))
        assign = F.softmax(sim / self.tau, dim=-1)
        z_proto = torch.matmul(assign, prototypes)
        delta = z_proto - z

        if not return_extra:
            return z_proto, assign, delta

        eps = 1e-8
        k = max(int(prototypes.shape[0]), 1)
        assign_entropy = -(assign * torch.log(assign + eps)).sum(dim=-1)
        entropy_denom = float(torch.log(torch.tensor(float(k))).item()) if k > 1 else 1.0
        assign_confidence_entropy = 1.0 - (assign_entropy / max(entropy_denom, eps))
        assign_confidence_entropy = assign_confidence_entropy.clamp(0.0, 1.0).unsqueeze(-1)

        top2 = torch.topk(assign, k=min(2, k), dim=-1).values
        top1_prob = top2[..., 0]
        top2_prob = top2[..., 1] if top2.shape[-1] > 1 else torch.zeros_like(top1_prob)
        assign_confidence_top2 = (top1_prob - top2_prob).clamp(0.0, 1.0).unsqueeze(-1)

        l2_dist = torch.cdist(z.reshape(-1, z.shape[-1]), prototypes, p=2).reshape(z.shape[0], z.shape[1], -1)
        min_proto_dist = l2_dist.min(dim=-1).values

        extra = {
            "assign_entropy": assign_entropy,
            "assign_confidence_entropy": assign_confidence_entropy,
            "assign_confidence_top2": assign_confidence_top2,
            "min_proto_l2_dist": min_proto_dist,
            "min_proto_dist": min_proto_dist,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "cosine_sim": sim,
        }
        return z_proto, assign, delta, extra


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
        correction_gate_enable: bool = False,
        correction_gate_mode: str = "entropy",
        correction_gate_min: float = 0.05,
        correction_gate_max: float = 1.0,
        correction_gate_detach: bool = True,
    ):
        super().__init__()
        self.lambda_node = lambda_node
        self.lambda_patch = lambda_patch
        self.use_node_correction = use_node_correction
        self.use_patch_correction = use_patch_correction
        self.fusion_mode = fusion_mode
        self.correction_gate_enable = correction_gate_enable
        self.correction_gate_mode = correction_gate_mode
        self.correction_gate_min = correction_gate_min
        self.correction_gate_max = correction_gate_max
        self.correction_gate_detach = correction_gate_detach

        self.linear_fuse = None
        if fusion_mode == "linear_fuse":
            self.linear_fuse = nn.Linear(latent_dim * 3, latent_dim)

    def _resolve_node_gate(self, node_extra: dict, z_fused: torch.Tensor):
        default_gate = torch.ones(z_fused.shape[0], z_fused.shape[1], 1, device=z_fused.device, dtype=z_fused.dtype)
        if not self.correction_gate_enable or node_extra is None:
            return default_gate

        if self.correction_gate_mode == "top2":
            gate = node_extra.get("assign_confidence_top2")
        else:
            gate = node_extra.get("assign_confidence_entropy")

        if gate is None:
            return default_gate

        gate = gate.clamp(self.correction_gate_min, self.correction_gate_max)
        if self.correction_gate_detach:
            gate = gate.detach()
        return gate

    def forward(
        self,
        z_fused: torch.Tensor,
        node_delta: torch.Tensor,
        patch_global_delta: torch.Tensor,
        z_node_proto: torch.Tensor,
        node_extra: dict = None,
    ):
        node_gate = self._resolve_node_gate(node_extra=node_extra, z_fused=z_fused)

        if self.fusion_mode == "linear_fuse":
            node_guided = z_fused + self.lambda_node * node_gate * node_delta if self.use_node_correction else z_fused
            patch_guided = patch_global_delta if self.use_patch_correction else torch.zeros_like(z_fused)
            z_cat = torch.cat([z_fused, node_guided, patch_guided], dim=-1)
            return self.linear_fuse(z_cat), node_gate

        z_corrected = z_fused
        if self.use_node_correction:
            z_corrected = z_corrected + self.lambda_node * node_gate * node_delta
        if self.use_patch_correction:
            z_corrected = z_corrected + self.lambda_patch * patch_global_delta
        return z_corrected, node_gate


class NodePrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)
        self.matcher = PrototypeMatcher(tau=tau)

    def forward(self, z_fused: torch.Tensor, return_extra: bool = False):
        return self.matcher(z_fused, self.prototypes, return_extra=return_extra)


class PatchPrototypeBank(nn.Module):
    def __init__(self, latent_dim: int, num_prototypes: int, tau: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.02)
        self.matcher = PrototypeMatcher(tau=tau)

    def forward(self, z_patch: torch.Tensor, return_extra: bool = False):
        return self.matcher(z_patch, self.prototypes, return_extra=return_extra)


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
        correction_gate_enable: bool = False,
        correction_gate_mode: str = "entropy",
        correction_gate_min: float = 0.05,
        correction_gate_max: float = 1.0,
        correction_gate_detach: bool = True,
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
            correction_gate_enable=correction_gate_enable,
            correction_gate_mode=correction_gate_mode,
            correction_gate_min=correction_gate_min,
            correction_gate_max=correction_gate_max,
            correction_gate_detach=correction_gate_detach,
        )

        self.register_buffer("node_proto_prior", torch.full((num_node_prototypes,), 1.0 / max(1, num_node_prototypes)))
        self.register_buffer("node_proto_radius", torch.ones(num_node_prototypes))

    def initialize_node_prototypes(self, centers: torch.Tensor, prior: torch.Tensor = None, radius: torch.Tensor = None):
        if self.node_bank is None:
            return
        with torch.no_grad():
            self.node_bank.prototypes.copy_(centers.to(self.node_bank.prototypes.device, dtype=self.node_bank.prototypes.dtype))
            k = centers.shape[0]
            prior_t = prior if prior is not None else torch.full((k,), 1.0 / max(k, 1), device=centers.device)
            radius_t = radius if radius is not None else torch.ones(k, device=centers.device)
            self.node_proto_prior = prior_t.to(self.node_bank.prototypes.device, dtype=self.node_bank.prototypes.dtype)
            self.node_proto_radius = radius_t.to(self.node_bank.prototypes.device, dtype=self.node_bank.prototypes.dtype)

    def get_node_prototypes(self):
        if self.node_bank is None:
            return None
        return self.node_bank.prototypes

    def get_node_prior(self):
        return self.node_proto_prior if self.node_bank is not None else None

    def get_node_radius(self):
        return self.node_proto_radius if self.node_bank is not None else None

    def get_node_prototype_pairwise_stats(self):
        prototypes = self.get_node_prototypes()
        if prototypes is None or prototypes.shape[0] < 2:
            return {
                "prototype_pairwise_distance_mean": 0.0,
                "prototype_pairwise_distance_min": 0.0,
                "prototype_pairwise_cosine_mean": 1.0,
                "prototype_pairwise_cosine_max": 1.0,
            }
        with torch.no_grad():
            dist = torch.cdist(prototypes, prototypes, p=2)
            mask = ~torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
            dist_vals = dist[mask]

            p_norm = F.normalize(prototypes, p=2, dim=-1)
            cos = torch.matmul(p_norm, p_norm.transpose(0, 1))
            cos_vals = cos[mask]
            return {
                "prototype_pairwise_distance_mean": float(dist_vals.mean().item()),
                "prototype_pairwise_distance_min": float(dist_vals.min().item()),
                "prototype_pairwise_cosine_mean": float(cos_vals.mean().item()),
                "prototype_pairwise_cosine_max": float(cos_vals.max().item()),
            }

    def match_to_node_prototypes(self, z: torch.Tensor, return_extra: bool = False):
        if self.node_bank is None:
            if return_extra:
                return None, None, None, None
            return None, None, None
        return self.node_bank.matcher(z, self.node_bank.prototypes, return_extra=return_extra)

    def forward(self, x: torch.Tensor, z_fused: torch.Tensor):
        batch_size, num_nodes, latent_dim = z_fused.shape

        node_assign = None
        node_delta = torch.zeros_like(z_fused)
        z_node_proto = None
        node_extra = None

        patch_assign = None
        z_patch = None
        z_patch_proto = None
        patch_delta = None
        patch_global_delta = torch.zeros_like(z_fused)

        if self.use_node_prototype and self.node_bank is not None:
            z_node_proto, node_assign, node_delta, node_extra = self.node_bank(z_fused, return_extra=True)

        if self.use_patch_prototype and self.patch_embedder is not None and self.patch_bank is not None:
            z_patch, _ = self.patch_embedder(x)
            z_patch_proto, patch_assign, patch_delta = self.patch_bank(z_patch)
            patch_global = patch_delta.mean(dim=1)
            patch_global_delta = patch_global.unsqueeze(1).expand(batch_size, num_nodes, latent_dim)

        z_corrected, node_gate = self.correction(
            z_fused=z_fused,
            node_delta=node_delta,
            patch_global_delta=patch_global_delta,
            z_node_proto=z_node_proto,
            node_extra=node_extra,
        )

        pairwise_stats = self.get_node_prototype_pairwise_stats()

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
            "node_assign_entropy": None if node_extra is None else node_extra.get("assign_entropy").detach(),
            "node_assign_confidence": None if node_extra is None else node_extra.get("assign_confidence_entropy").detach(),
            "node_correction_gate": node_gate.detach(),
            "node_min_proto_dist": None if node_extra is None else node_extra.get("min_proto_dist").detach(),
            "prototype_pairwise_distance_mean": pairwise_stats["prototype_pairwise_distance_mean"],
            "prototype_pairwise_distance_min": pairwise_stats["prototype_pairwise_distance_min"],
            "prototype_pairwise_cosine_mean": pairwise_stats["prototype_pairwise_cosine_mean"],
            "prototype_pairwise_cosine_max": pairwise_stats["prototype_pairwise_cosine_max"],
        }
