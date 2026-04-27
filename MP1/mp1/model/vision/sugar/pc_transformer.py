import random
import numpy as np
import torch
import torch.nn as nn

from .transformer import SelfAttentionBlock, CrossAttentionBlock
from .point_ops import PointGroup


class PCGroupEncoder(nn.Module):
    """Embeds each point group into a fixed-size token."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_size, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, output_size, 1),
        )

    def forward(self, point_groups):
        """
        point_groups: (B, G, N, C)
        returns: (B, G, output_size)
        """
        B, G, N, C = point_groups.shape
        x = point_groups.reshape(B * G, N, C).transpose(2, 1)  # (B*G, C, N)
        feat = self.first_conv(x)                               # (B*G, 256, N)
        feat_global = feat.max(dim=2, keepdim=True)[0]          # (B*G, 256, 1)
        feat = torch.cat([feat_global.expand(-1, -1, N), feat], dim=1)  # (B*G, 512, N)
        feat = self.second_conv(feat)                           # (B*G, output_size, N)
        feat_global = feat.max(dim=2)[0]                        # (B*G, output_size)
        return feat_global.reshape(B, G, self.output_size)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attn_input='post', reproduce_recon=False):
        super().__init__()
        self.reproduce_recon = reproduce_recon
        self.cross_attn_input = 'extra' if reproduce_recon else cross_attn_input
        assert self.cross_attn_input in ['pre', 'post', 'extra']

        self.encoder_block = SelfAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, reproduce_recon=reproduce_recon)
        self.decoder_block_sa = SelfAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, reproduce_recon=reproduce_recon)
        self.decoder_block_ca = CrossAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, reproduce_recon=reproduce_recon)

    def update_tgt_given_src(self, src, tgt, src_padded_mask=None, tgt_padded_mask=None,
                              skip_tgt_sa=False, detach_src=False):
        if not skip_tgt_sa:
            tgt = self.decoder_block_sa(tgt, tgt_padded_mask)
        tgt = self.decoder_block_ca(tgt, src, src_padded_mask=src_padded_mask, detach_src=detach_src)
        return tgt

    def forward(self, src, tgt=None, src_padded_mask=None, tgt_padded_mask=None,
                skip_tgt_sa=False, detach_src=False):
        src_out = self.encoder_block(src, padded_mask=src_padded_mask)
        if self.reproduce_recon:
            src_out, src_extra = src_out

        if self.cross_attn_input == 'post':
            ca_input = src_out
        elif self.cross_attn_input == 'pre':
            ca_input = src
        else:  # 'extra'
            ca_input = src_extra['norm1']

        if tgt is None:
            return src_out, None, ca_input

        if not skip_tgt_sa:
            tgt = self.decoder_block_sa(tgt, tgt_padded_mask)
        tgt = self.decoder_block_ca(tgt, ca_input, src_padded_mask=src_padded_mask, detach_src=detach_src)
        return src_out, tgt, ca_input


class MaskedPCTransformer(nn.Module):
    def __init__(
        self, hidden_size=384, num_heads=6, depth=12,
        input_size=6, num_groups=64, group_size=32,
        group_use_knn=True, group_radius=None,
        drop_path_rate=0.1, mask_ratio=0., mask_type='rand',
        cross_attn_input='post', cross_attn_layers=None, **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

        self.group_divider = PointGroup(
            num_groups=num_groups, group_size=group_size,
            knn=group_use_knn, radius=group_radius)
        self.encoder = PCGroupEncoder(input_size, hidden_size)
        self.point_pos_embedding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_size))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=hidden_size, num_heads=num_heads, mlp_ratio=4,
                  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                  drop_path=dpr[i], cross_attn_input=cross_attn_input)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(B, G, dtype=torch.bool, device=center.device)
        num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([np.zeros(G - num_mask), np.ones(num_mask)])
            np.random.shuffle(mask)
            overall_mask[i] = mask
        return torch.from_numpy(overall_mask).bool().to(center.device)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)
        mask_idx = []
        for points in center:
            index = random.randint(0, points.size(0) - 1)
            dist = torch.norm(points[index].reshape(1, 3) - points, p=2, dim=-1)
            idx = torch.argsort(dist)
            num_mask = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:num_mask]] = 1
            mask_idx.append(mask.bool())
        return torch.stack(mask_idx).to(center.device)

    def forward(self, pc_fts, query=None, query_pos=None, query_padded_mask=None,
                skip_tgt_sa=False, detach_src=False, return_multiscale_layers=None,
                return_ca_inputs=False, mask_pc=True):
        """
        pc_fts: (B, N, C)  first 3 dims are xyz
        returns dict with keys: pc_vis, query, centers, neighborhoods, pc_bool_masks
        """
        neighborhoods, centers = self.group_divider(pc_fts)

        if mask_pc:
            if self.mask_type == 'rand':
                pc_bool_masks = self._mask_center_rand(centers)
            else:
                pc_bool_masks = self._mask_center_block(centers)
        else:
            pc_bool_masks = torch.zeros(centers.shape[:2], dtype=torch.bool, device=centers.device)

        group_tokens = self.encoder(neighborhoods)           # (B, G, hidden)
        B, G, C = group_tokens.shape
        pc_vis = group_tokens[~pc_bool_masks].reshape(B, -1, C)
        centers_vis = centers[~pc_bool_masks].reshape(B, -1, 3)
        pos_vis = self.point_pos_embedding(centers_vis)

        multiscale_fts = [] if return_multiscale_layers is not None else None
        ca_inputs = [] if return_ca_inputs else None

        for kth, block in enumerate(self.blocks):
            if query is not None and query_pos is not None:
                query = query + query_pos
            pc_vis, query, ca_input = block(
                pc_vis + pos_vis, tgt=query,
                tgt_padded_mask=query_padded_mask,
                skip_tgt_sa=skip_tgt_sa, detach_src=detach_src)
            if return_ca_inputs:
                ca_inputs.append(ca_input)
            if kth == len(self.blocks) - 1:
                pc_vis = self.norm(pc_vis)
                if query is not None:
                    query = self.norm(query)
            if multiscale_fts is not None and (kth + 1) in return_multiscale_layers:
                multiscale_fts.append(pc_vis)

        outs = {
            'pc_vis': pc_vis,
            'query': query,
            'centers': centers,
            'neighborhoods': neighborhoods,
            'pc_bool_masks': pc_bool_masks,
        }
        if multiscale_fts is not None:
            outs['multiscale_pc_fts'] = torch.cat(multiscale_fts, dim=2)
        if ca_inputs is not None:
            outs['ca_inputs'] = ca_inputs
        return outs
