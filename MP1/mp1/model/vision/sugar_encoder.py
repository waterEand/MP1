import torch
import torch.nn as nn
from typing import Dict, List, Type
from termcolor import cprint

from mp1.model.vision.sugar import MaskedPCTransformer
from mp1.model.vision.pointnet_extractor import create_mlp


class SUGARExtractor(nn.Module):
    """
    Wraps SUGAR's MaskedPCTransformer as a point cloud feature extractor.

    Input : (B, N, 3) XYZ point cloud (any N).
    Output: (B, out_channels) global feature vector.

    Key design notes
    ----------------
    * The checkpoint was pretrained with input_size=6 (XYZ+RGB) and group_size=32.
      We keep input_size=6 but zero-pad RGB (RGB norms are small vs XYZ in the
      pretrained weights so this only reduces activation ~3%).
    * To match the pretrained group_size=32, points are upsampled to
      `target_npoints` (default 1024) before grouping.  With num_groups=32 and
      group_size=32 this reproduces exactly the pretraining local-patch scale.
    * When freeze=True the transformer is kept in eval mode at all times so that
      BatchNorm and DropPath behave deterministically during policy training.
    """

    def __init__(
        self,
        pretrained_path: str,
        num_groups: int = 32,
        group_size: int = 32,
        target_npoints: int = 1024,   # upsample input point cloud to this count
        hidden_size: int = 384,
        num_heads: int = 6,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        freeze: bool = False,
        finetune_last_n_layers: int = 0,
        out_channels: int = 64,
        projection_hidden_dim: int = 256,
    ):
        super().__init__()
        self.target_npoints = target_npoints

        self.transformer = MaskedPCTransformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=depth,
            input_size=6,
            num_groups=num_groups,
            group_size=group_size,
            drop_path_rate=drop_path_rate,
            mask_ratio=0.0,
            mask_type='rand',
            cross_attn_input='post',
        )

        self._load_pretrained(pretrained_path)

        if freeze:
            self._freeze_all()
            if finetune_last_n_layers > 0:
                self._unfreeze_last_n(finetune_last_n_layers, depth)
        else:
            cprint('[SUGARExtractor] Full fine-tuning (freeze=False)', 'cyan')

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden_dim),
            nn.GELU(),
            nn.Linear(projection_hidden_dim, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.out_channels = out_channels

    # ── weight loading ───────────────────────────────────────────────────────

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        prefix = 'mae_encoder.'
        encoder_sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        missing, unexpected = self.transformer.load_state_dict(encoder_sd, strict=False)
        cprint(f'[SUGARExtractor] Loaded {path}. Missing={len(missing)}, Unexpected={len(unexpected)}', 'cyan')
        if missing:
            cprint(f'  Missing keys (first 5): {missing[:5]}', 'yellow')

    # ── freeze helpers ───────────────────────────────────────────────────────

    def _freeze_all(self):
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self._frozen = True
        cprint('[SUGARExtractor] Encoder frozen (transformer will stay in eval mode)', 'cyan')

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_frozen', False):
            self.transformer.eval()
        return self

    def _unfreeze_last_n(self, n: int, depth: int):
        for i in range(depth - n, depth):
            for p in self.transformer.blocks[i].parameters():
                p.requires_grad_(True)
        for p in self.transformer.norm.parameters():
            p.requires_grad_(True)
        cprint(f'[SUGARExtractor] Unfroze last {n} transformer blocks', 'cyan')

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, N, 3)
        returns: (B, out_channels)
        """
        B, N, _ = xyz.shape

        # Upsample to target_npoints so group_size=32 matches pretraining scale.
        # Training: random sampling with replacement (free data augmentation).
        # Eval: deterministic tiling to keep inference reproducible.
        if N < self.target_npoints:
            if self.training:
                idx = torch.randint(0, N, (B, self.target_npoints), device=xyz.device)
                xyz = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), idx]
            else:
                repeats = (self.target_npoints + N - 1) // N
                xyz = xyz.repeat(1, repeats, 1)[:, :self.target_npoints, :]

        # Zero-pad color channels (RGB weight norms are small; ~3% activation impact)
        pc_fts = torch.zeros(B, self.target_npoints, 6, device=xyz.device, dtype=xyz.dtype)
        pc_fts[..., :3] = xyz

        outs = self.transformer(pc_fts, mask_pc=False)
        pc_vis = outs['pc_vis']                  # (B, num_groups, hidden_size)
        global_feat = pc_vis.max(dim=1).values   # (B, hidden_size)
        return self.projection(global_feat)       # (B, out_channels)


class MP1SugarEncoder(nn.Module):
    """
    Drop-in replacement for MP1Encoder that uses SUGAR as the point cloud backbone.
    Same interface: forward(observations) -> Tensor, output_shape() -> int.
    """

    def __init__(
        self,
        observation_space: Dict,
        out_channel: int = 64,
        state_mlp_size: tuple = (64, 64),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        sugar_encoder_cfg: dict = None,
    ):
        super().__init__()
        self.point_cloud_key = 'point_cloud'
        self.state_key = 'agent_pos'

        if sugar_encoder_cfg is None:
            sugar_encoder_cfg = {}
        sugar_encoder_cfg.setdefault('out_channels', out_channel)

        self.extractor = SUGARExtractor(**sugar_encoder_cfg)
        pc_feat_dim = self.extractor.out_channels

        state_dim = observation_space[self.state_key][0]
        if len(state_mlp_size) == 0:
            raise ValueError('state_mlp_size must not be empty')
        net_arch = list(state_mlp_size[:-1])
        state_out_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(
            *create_mlp(state_dim, state_out_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = pc_feat_dim + state_out_dim
        cprint(f'[MP1SugarEncoder] pc_feat={pc_feat_dim}, state_feat={state_out_dim}, '
               f'total={self.n_output_channels}', 'red')

    def forward(self, observations: Dict) -> torch.Tensor:
        xyz = observations[self.point_cloud_key][..., :3]
        pc_feat = self.extractor(xyz)
        state_feat = self.state_mlp(observations[self.state_key])
        return torch.cat([pc_feat, state_feat], dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels
