import torch
import torch.nn as nn
from typing import Dict, List, Type
from termcolor import cprint

from mp1.model.vision.sugar import MaskedPCTransformer
from mp1.model.vision.pointnet_extractor import create_mlp


class SUGARExtractor(nn.Module):
    """
    Wraps SUGAR's MaskedPCTransformer as a point cloud feature extractor.

    Two output modes (controlled by `per_token_output`):
    ─────────────────────────────────────────────────
    per_token_output=False  (FiLM / global):
        global max-pool over tokens → MLP projection → (B, out_channels)

    per_token_output=True   (cross-attention / per-token):
        all tokens → linear projection → (B, num_groups, token_dim)
        `out_channels` is used as token_dim; `projection_hidden_dim` is ignored.
        Recommended for use with condition_type: cross_attention in the UNet.

    Input:  (B, N, 3) XYZ point cloud (N ≥ group_size; upsampled to target_npoints internally)
    """

    def __init__(
        self,
        pretrained_path: str,
        num_groups: int = 32,
        group_size: int = 32,
        target_npoints: int = 1024,
        hidden_size: int = 384,
        num_heads: int = 6,
        depth: int = 12,
        drop_path_rate: float = 0.1,
        freeze: bool = False,
        finetune_last_n_layers: int = 0,
        out_channels: int = 64,
        projection_hidden_dim: int = 256,
        per_token_output: bool = False,
    ):
        super().__init__()
        self.target_npoints = target_npoints
        self.per_token_output = per_token_output
        self.out_channels = out_channels

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

        if per_token_output:
            # Lightweight linear projection at token level (like FPVNet: 384 → token_dim)
            self.projection = nn.Linear(hidden_size, out_channels)
            cprint(f'[SUGARExtractor] per-token mode: {hidden_size} → {out_channels} per token', 'cyan')
        else:
            # Global: max-pool → MLP
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, projection_hidden_dim),
                nn.GELU(),
                nn.Linear(projection_hidden_dim, out_channels),
                nn.LayerNorm(out_channels),
            )

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
        cprint('[SUGARExtractor] Encoder frozen (transformer stays in eval mode)', 'cyan')

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
        Returns:
            per_token_output=False → (B, out_channels)
            per_token_output=True  → (B, num_groups, out_channels)
        """
        B, N, _ = xyz.shape

        # Upsample: train=random sampling (augmentation), eval=deterministic tiling
        if N < self.target_npoints:
            if self.training:
                idx = torch.randint(0, N, (B, self.target_npoints), device=xyz.device)
                xyz = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), idx]
            else:
                repeats = (self.target_npoints + N - 1) // N
                xyz = xyz.repeat(1, repeats, 1)[:, :self.target_npoints, :]

        pc_fts = torch.zeros(B, self.target_npoints, 6, device=xyz.device, dtype=xyz.dtype)
        pc_fts[..., :3] = xyz

        outs = self.transformer(pc_fts, mask_pc=False)
        pc_vis = outs['pc_vis']  # (B, num_groups, hidden_size)

        if self.per_token_output:
            return self.projection(pc_vis)  # (B, num_groups, out_channels)
        else:
            global_feat = pc_vis.max(dim=1).values  # (B, hidden_size)
            return self.projection(global_feat)      # (B, out_channels)


class MP1SugarEncoder(nn.Module):
    """
    Drop-in replacement for MP1Encoder using SUGAR as the point cloud backbone.

    Two modes, selected by `per_token_output` in `sugar_encoder_cfg`:

    per_token_output=False  (FiLM conditioning, default):
        Returns flat vector (B, pc_feat + state_feat).
        output_shape() = scalar int.  Identical interface to MP1Encoder.

    per_token_output=True  (cross-attention conditioning):
        Returns token sequence (B, num_groups + 1, token_dim).
        The +1 token encodes the agent state.
        output_shape() = token_dim (per-token embedding dim).
        Use with condition_type: cross_attention in the policy.
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
        self.per_token_output = self.extractor.per_token_output
        token_dim = self.extractor.out_channels

        state_dim = observation_space[self.state_key][0]

        if self.per_token_output:
            # Per-token mode: state → single token of same dim
            # MLP: state_dim → token_dim
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.n_output_channels = token_dim  # per-token dim (for UNet cross-attention)
            cprint(f'[MP1SugarEncoder] per-token mode: token_dim={token_dim}', 'red')
        else:
            # Global mode: state MLP → concat with pc feat
            if len(state_mlp_size) == 0:
                raise ValueError('state_mlp_size must not be empty')
            net_arch = list(state_mlp_size[:-1])
            state_out_dim = state_mlp_size[-1]
            self.state_mlp = nn.Sequential(
                *create_mlp(state_dim, state_out_dim, net_arch, state_mlp_activation_fn))
            self.n_output_channels = token_dim + state_out_dim
            cprint(f'[MP1SugarEncoder] global mode: pc={token_dim}, state={state_out_dim}, '
                   f'total={self.n_output_channels}', 'red')

    def forward(self, observations: Dict) -> torch.Tensor:
        xyz = observations[self.point_cloud_key][..., :3]

        if self.per_token_output:
            pc_tokens = self.extractor(xyz)  # (B, G, token_dim)
            state_tok = self.state_mlp(observations[self.state_key]).unsqueeze(1)  # (B, 1, token_dim)
            return torch.cat([pc_tokens, state_tok], dim=1)  # (B, G+1, token_dim)
        else:
            pc_feat = self.extractor(xyz)
            state_feat = self.state_mlp(observations[self.state_key])
            return torch.cat([pc_feat, state_feat], dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels
