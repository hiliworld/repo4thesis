import torch
import torch.nn as nn

from src.models.encoders import LNTMultiSlotEncoder, LNT_Conv_Encoder, SlotAggregator
from src.models.layers import SpatialAttentionLayer
from src.models.prototypes import PrototypeFusionModule
from src.models.segment_encoder import LearnableSegmentEncoder


class ModelOutput(dict):
    """Dict-like output with backward-compatible tuple-style access."""

    tuple_keys = ("pred", "recon", "z_corrected")

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(self.tuple_keys[key])
        return super().__getitem__(key)

    def __iter__(self):
        for key in self.tuple_keys:
            yield super().__getitem__(key)

    def __len__(self):
        return len(self.tuple_keys)


class MyFinalModel(nn.Module):
    def __init__(self, config):
        super(MyFinalModel, self).__init__()

        self.num_nodes = config["dataset"]["input_dim"]
        self.window_size = config["dataset"]["window_size"]
        self.hidden_dim = int(config["model"].get("hidden_dim", 64))

        model_cfg = config.get("model", {})
        self.backbone_version = str(model_cfg.get("backbone_version", "legacy")).lower()
        self.lnt_num_slots = int(model_cfg.get("lnt_num_slots", 2))
        self.lnt_slot_agg = str(model_cfg.get("lnt_slot_agg", "gated_sum"))
        self.gat_mode = str(model_cfg.get("gat_mode", "legacy")).lower()
        self.local_global_fusion = str(model_cfg.get("local_global_fusion", "sum")).lower()
        self.return_slot_debug = bool(model_cfg.get("return_slot_debug", False))
        self.use_slot_pos_embedding = bool(model_cfg.get("use_slot_pos_embedding", True))

        prototype_cfg = model_cfg.get("prototype", {})
        prototype_v2_cfg = config.get("prototype_v2", {})
        correction_gate_cfg = prototype_v2_cfg.get("correction_gate", {})
        self.use_prototype_fusion = bool(model_cfg.get("use_prototype_fusion", False))

        if self.backbone_version == "lnt_v2_slotwise":
            self.metric_encoder_v2 = LNTMultiSlotEncoder(
                input_dim=1,
                z_dim=self.hidden_dim,
                num_slots=self.lnt_num_slots,
                use_slot_pos_embedding=self.use_slot_pos_embedding,
            )
            self.metric_encoder = None
            self.local_slot_aggregator = SlotAggregator(
                hidden_dim=self.hidden_dim,
                num_slots=self.lnt_num_slots,
                mode=self.lnt_slot_agg,
            )
            self.global_slot_aggregator = SlotAggregator(
                hidden_dim=self.hidden_dim,
                num_slots=self.lnt_num_slots,
                mode=self.lnt_slot_agg,
            )
        else:
            self.metric_encoder_v2 = None
            self.metric_encoder = LNT_Conv_Encoder(
                input_dim=1,
                z_dim=self.hidden_dim,
                pool_bins=model_cfg.get("lnt_pool_bins", 1),
            )
            self.local_slot_aggregator = None
            self.global_slot_aggregator = None

        self.gat_layer = SpatialAttentionLayer(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.2,
        )

        self.local_global_gate = None
        if self.local_global_fusion == "gated":
            self.local_global_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )

        anomaly_space_cfg = config.get("anomaly_space", {})
        self.use_learnable_segment_encoder = bool(anomaly_space_cfg.get("use_learnable_segment_encoder", False))
        self.segment_encoder = None
        if self.use_learnable_segment_encoder:
            self.segment_encoder = LearnableSegmentEncoder(
                hidden_dim=self.hidden_dim,
                repr_dim=int(anomaly_space_cfg.get("segment_repr_dim", 64)),
                score_dim=4,
            )

        self.prototype_fusion = None
        if self.use_prototype_fusion:
            use_node_prototype = prototype_cfg.get("use_node_prototype", True)
            use_patch_prototype = prototype_cfg.get("use_patch_prototype", True)
            self.prototype_fusion = PrototypeFusionModule(
                input_dim=self.num_nodes,
                latent_dim=self.hidden_dim,
                use_node_prototype=use_node_prototype,
                use_patch_prototype=use_patch_prototype,
                use_node_correction=(prototype_cfg.get("use_node_correction", True) and use_node_prototype),
                use_patch_correction=(prototype_cfg.get("use_patch_correction", True) and use_patch_prototype),
                num_node_prototypes=prototype_cfg.get("num_node_prototypes", 8),
                num_patch_prototypes=prototype_cfg.get("num_patch_prototypes", 8),
                tau_node=prototype_cfg.get("tau_node", 0.1),
                tau_patch=prototype_cfg.get("tau_patch", 0.1),
                lambda_node=prototype_cfg.get("lambda_node", 0.5),
                lambda_patch=prototype_cfg.get("lambda_patch", 0.3),
                patch_size=prototype_cfg.get("patch_size", 10),
                fusion_mode=prototype_cfg.get("fusion_mode", "residual_add"),
                correction_gate_enable=bool(correction_gate_cfg.get("enable", False)),
                correction_gate_mode=str(correction_gate_cfg.get("mode", "entropy")),
                correction_gate_min=float(correction_gate_cfg.get("min_gate", 0.05)),
                correction_gate_max=float(correction_gate_cfg.get("max_gate", 1.0)),
                correction_gate_detach=bool(correction_gate_cfg.get("detach_gate", True)),
            )

        self.pred_head = nn.Sequential(nn.Linear(self.hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.recon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.window_size // 2),
            nn.ReLU(),
            nn.Linear(self.window_size // 2, self.window_size),
        )

        self.log_encoder = None

    def _fuse_local_global(self, z_local, z_global):
        # z_local: [B, N, D], z_global: [B, N, D]
        if self.local_global_fusion == "gated" and self.local_global_gate is not None:
            gate = self.local_global_gate(torch.cat([z_local, z_global], dim=-1))  # [B, N, D]
            z_fused = gate * z_local + (1.0 - gate) * z_global
            return z_fused, gate

        z_fused = z_local + z_global
        gate = torch.ones_like(z_local)
        return z_fused, gate

    def _encode_legacy(self, x):
        # x: [B, T, N]
        z_local = self.metric_encoder(x)  # [B, N, D]
        z_global, _ = self.gat_layer(z_local)  # [B, N, D]
        z_fused, lg_gate = self._fuse_local_global(z_local, z_global)

        # Compatibility placeholders for slotwise outputs
        z_local_slots = z_local.unsqueeze(2)  # [B, N, 1, D]
        z_global_slots = z_global.unsqueeze(2)  # [B, N, 1, D]
        slot_weights_local = torch.ones_like(z_local_slots[:, :, :, :1])
        slot_weights_global = torch.ones_like(z_global_slots[:, :, :, :1])
        return z_local, z_global, z_fused, z_local_slots, z_global_slots, slot_weights_local, slot_weights_global, lg_gate

    def _encode_slotwise(self, x):
        # x: [B, T, N]
        z_local_slots = self.metric_encoder_v2(x)  # [B, N, S, D]

        if self.gat_mode == "slotwise_shared":
            global_slots = []
            for slot_idx in range(z_local_slots.size(2)):
                z_slot = z_local_slots[:, :, slot_idx, :]  # [B, N, D]
                z_slot_global, _ = self.gat_layer(z_slot)  # [B, N, D]
                global_slots.append(z_slot_global)
            z_global_slots = torch.stack(global_slots, dim=2)  # [B, N, S, D]
        else:
            z_local, slot_weights_local = self.local_slot_aggregator(z_local_slots)
            z_global, _ = self.gat_layer(z_local)
            z_global_slots = z_global.unsqueeze(2).expand_as(z_local_slots)
            slot_weights_global = slot_weights_local
            z_fused, lg_gate = self._fuse_local_global(z_local, z_global)
            return z_local, z_global, z_fused, z_local_slots, z_global_slots, slot_weights_local, slot_weights_global, lg_gate

        z_local, slot_weights_local = self.local_slot_aggregator(z_local_slots)
        z_global, slot_weights_global = self.global_slot_aggregator(z_global_slots)
        z_fused, lg_gate = self._fuse_local_global(z_local, z_global)
        return z_local, z_global, z_fused, z_local_slots, z_global_slots, slot_weights_local, slot_weights_global, lg_gate

    def _build_window_repr(self, outputs):
        if self.segment_encoder is None:
            return None
        score_stats = outputs.get("score_stats")
        return self.segment_encoder.encode_window(
            z_local_slots=outputs["z_local_slots"],
            z_global_slots=outputs["z_global_slots"],
            z_corrected=outputs["z_corrected"],
            score_stats=score_stats,
        )

    def forward(self, x):
        """x: [B, T, N]"""
        if self.backbone_version == "lnt_v2_slotwise":
            (
                z_local,
                z_global,
                z_fused,
                z_local_slots,
                z_global_slots,
                slot_weights_local,
                slot_weights_global,
                local_global_gate,
            ) = self._encode_slotwise(x)
        else:
            (
                z_local,
                z_global,
                z_fused,
                z_local_slots,
                z_global_slots,
                slot_weights_local,
                slot_weights_global,
                local_global_gate,
            ) = self._encode_legacy(x)

        proto_outputs = {
            "z_corrected": z_fused,
            "node_assign": None,
            "patch_assign": None,
            "node_proto": None,
            "patch_proto": None,
            "node_delta": None,
            "patch_delta": None,
            "patch_global_delta": None,
            "z_patch": None,
        }
        if self.use_prototype_fusion and self.prototype_fusion is not None:
            proto_outputs = self.prototype_fusion(x, z_fused)

        z_corrected = proto_outputs["z_corrected"]

        pred_raw = self.pred_head(z_fused).squeeze(-1)  # [B, N]
        recon_raw = self.recon_head(z_fused).permute(0, 2, 1)  # [B, T, N]
        pred_corrected = self.pred_head(z_corrected).squeeze(-1)  # [B, N]
        recon_corrected = self.recon_head(z_corrected).permute(0, 2, 1)  # [B, T, N]

        base = ModelOutput(
            {
                "pred": pred_corrected,
                "recon": recon_corrected,
                "pred_raw": pred_raw,
                "recon_raw": recon_raw,
                "pred_corrected": pred_corrected,
                "recon_corrected": recon_corrected,
                "z_local": z_local,
                "z_global": z_global,
                "z_fused": z_fused,
                "z_corrected": z_corrected,
                "z_local_slots": z_local_slots,
                "z_global_slots": z_global_slots,
                "slot_weights_local": slot_weights_local,
                "slot_weights_global": slot_weights_global,
                "local_global_gate": local_global_gate,
                "node_assign": proto_outputs["node_assign"],
                "patch_assign": proto_outputs["patch_assign"],
                "node_proto_latent": proto_outputs["node_proto"],
                "patch_proto_latent": proto_outputs["patch_proto"],
                "node_delta": proto_outputs["node_delta"],
                "patch_delta": proto_outputs["patch_delta"],
                "patch_global_delta": proto_outputs["patch_global_delta"],
                "z_patch": proto_outputs["z_patch"],
                "node_assign_entropy": proto_outputs.get("node_assign_entropy"),
                "node_assign_confidence": proto_outputs.get("node_assign_confidence"),
                "node_correction_gate": proto_outputs.get("node_correction_gate"),
                "node_min_proto_dist": proto_outputs.get("node_min_proto_dist"),
                "prototype_pairwise_distance_mean": proto_outputs.get("prototype_pairwise_distance_mean"),
                "prototype_pairwise_distance_min": proto_outputs.get("prototype_pairwise_distance_min"),
                "prototype_pairwise_cosine_mean": proto_outputs.get("prototype_pairwise_cosine_mean"),
                "prototype_pairwise_cosine_max": proto_outputs.get("prototype_pairwise_cosine_max"),
                "score_stats": None,
                "window_repr": None,
            }
        )

        base["window_repr"] = self._build_window_repr(base)

        if not self.return_slot_debug:
            base.pop("slot_weights_local", None)
            base.pop("slot_weights_global", None)
            base.pop("local_global_gate", None)

        return base
