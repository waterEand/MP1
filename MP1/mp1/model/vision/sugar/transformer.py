import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth per sample (drop entire residual branch)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob).div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, src, padded_mask=None):
        B, N, C = src.shape
        qkv = self.qkv(src).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if padded_mask is not None:
            attn.masked_fill_(padded_mask.unsqueeze(1).unsqueeze(2), -float('inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        src_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        src_out = self.proj(src_out)
        src_out = self.proj_drop(src_out)
        return src_out


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_k=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim_k if dim_k is not None else dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tgt, src, src_padded_mask=None):
        B, N_src, C = src.shape
        kv = self.kv(src).reshape(B, N_src, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N_tgt, _ = tgt.shape
        q = self.q(tgt).reshape(B, N_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if src_padded_mask is not None:
            attn.masked_fill_(src_padded_mask.unsqueeze(1).unsqueeze(2), -float('inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        tgt_out = (attn @ v).transpose(1, 2).reshape(B, N_tgt, C)
        tgt_out = self.proj(tgt_out)
        tgt_out = self.proj_drop(tgt_out)
        return tgt_out


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 reproduce_recon=False):
        super().__init__()
        self.reproduce_recon = reproduce_recon
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop)

    def forward(self, src, padded_mask=None):
        if self.reproduce_recon:
            src_norm1 = self.norm1(src)
            src = src_norm1 + self.drop_path(self.attn(src_norm1, padded_mask))
            src = self.norm2(src)
            src = src + self.drop_path(self.mlp(src))
            return src, {'norm1': src_norm1}
        else:
            src = src + self.drop_path(self.attn(self.norm1(src), padded_mask))
            src = src + self.drop_path(self.mlp(self.norm2(src)))
            return src


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_k=None, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 reproduce_recon=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.cross_attn = CrossAttention(dim, dim_k=dim_k, num_heads=num_heads, qkv_bias=qkv_bias,
                                         qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, tgt, src, src_padded_mask=None, detach_src=False):
        if detach_src:
            src = src.detach()
        tgt = tgt + self.drop_path(self.cross_attn(self.norm1(tgt), src, src_padded_mask))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return tgt
