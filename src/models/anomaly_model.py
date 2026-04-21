import torch.nn as nn

# 导入自定义模块
from src.models.encoders import LNT_Conv_Encoder
from src.models.layers import SpatialAttentionLayer
from src.models.prototypes import PrototypeFusionModule


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

        # === 1. 基础配置 ===
        self.num_nodes = config['dataset']['input_dim']
        self.window_size = config['dataset']['window_size']
        self.hidden_dim = config['model'].get('hidden_dim', 64)

        prototype_cfg = config['model'].get('prototype', {})
        self.use_prototype_fusion = config['model'].get('use_prototype_fusion', False)

        # === 2. 核心组件初始化 ===
        self.metric_encoder = LNT_Conv_Encoder(
            input_dim=1,
            z_dim=self.hidden_dim,
            pool_bins=config['model'].get('lnt_pool_bins', 1),
        )
        self.gat_layer = SpatialAttentionLayer(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.2,
        )

        self.prototype_fusion = None
        if self.use_prototype_fusion:
            use_node_prototype = prototype_cfg.get('use_node_prototype', True)
            use_patch_prototype = prototype_cfg.get('use_patch_prototype', True)
            self.prototype_fusion = PrototypeFusionModule(
                input_dim=self.num_nodes,
                latent_dim=self.hidden_dim,
                use_node_prototype=use_node_prototype,
                use_patch_prototype=use_patch_prototype,
                use_node_correction=(prototype_cfg.get('use_node_correction', True) and use_node_prototype),
                use_patch_correction=(prototype_cfg.get('use_patch_correction', True) and use_patch_prototype),
                num_node_prototypes=prototype_cfg.get('num_node_prototypes', 8),
                num_patch_prototypes=prototype_cfg.get('num_patch_prototypes', 8),
                tau_node=prototype_cfg.get('tau_node', 0.1),
                tau_patch=prototype_cfg.get('tau_patch', 0.1),
                lambda_node=prototype_cfg.get('lambda_node', 0.5),
                lambda_patch=prototype_cfg.get('lambda_patch', 0.3),
                patch_size=prototype_cfg.get('patch_size', 10),
                fusion_mode=prototype_cfg.get('fusion_mode', 'residual_add'),
            )

        # === 3. 任务头 ===
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.window_size // 2),
            nn.ReLU(),
            nn.Linear(self.window_size // 2, self.window_size),
        )

        self.log_encoder = None

    def forward(self, x):
        """x: [Batch, Window, Nodes]"""
        z_local = self.metric_encoder(x)
        z_global, _ = self.gat_layer(z_local)
        z_fused = z_local + z_global

        proto_outputs = {
            'z_corrected': z_fused,
            'node_assign': None,
            'patch_assign': None,
            'node_proto': None,
            'patch_proto': None,
            'node_delta': None,
            'patch_delta': None,
            'patch_global_delta': None,
            'z_patch': None,
        }
        if self.use_prototype_fusion and self.prototype_fusion is not None:
            proto_outputs = self.prototype_fusion(x, z_fused)

        z_corrected = proto_outputs['z_corrected']
        pred_raw = self.pred_head(z_fused).squeeze(-1)
        recon_raw = self.recon_head(z_fused).permute(0, 2, 1)
        pred_corrected = self.pred_head(z_corrected).squeeze(-1)
        recon_corrected = self.recon_head(z_corrected).permute(0, 2, 1)

        return ModelOutput({
            # 为了兼容现有训练/测试脚本，默认 pred/recon 走 corrected 分支
            'pred': pred_corrected,
            'recon': recon_corrected,
            # 新增双路输出：raw vs corrected
            'pred_raw': pred_raw,
            'recon_raw': recon_raw,
            'pred_corrected': pred_corrected,
            'recon_corrected': recon_corrected,
            'z_fused': z_fused,
            'z_corrected': z_corrected,
            'node_assign': proto_outputs['node_assign'],
            'patch_assign': proto_outputs['patch_assign'],
            'node_proto_latent': proto_outputs['node_proto'],
            'patch_proto_latent': proto_outputs['patch_proto'],
            'node_delta': proto_outputs['node_delta'],
            'patch_delta': proto_outputs['patch_delta'],
            'patch_global_delta': proto_outputs['patch_global_delta'],
            'z_patch': proto_outputs['z_patch'],
        })
