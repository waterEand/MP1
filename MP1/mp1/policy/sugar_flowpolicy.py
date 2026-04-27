import sys
sys.path.append('MP1/mp1')
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from termcolor import cprint

from mp1.sde_lib import ConsistencyFM
from mp1.model.common.normalizer import LinearNormalizer
from mp1.policy.base_policy import BasePolicy
from mp1.model.mean.conditional_unet1d import ConditionalUnet1D
from mp1.model.mean.mask_generator import LowdimMaskGenerator
from mp1.common.pytorch_util import dict_apply
from mp1.common.model_util import print_params
from mp1.model.vision.pointnet_extractor import MP1Encoder
from mp1.model.vision.sugar_encoder import MP1SugarEncoder

import warnings
warnings.filterwarnings("ignore")


class SugarFlowPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        # encoder selection
        encoder_type: str = "mlp",          # "mlp" or "sugar"
        encoder_output_dim: int = 256,
        # mlp encoder params (used when encoder_type="mlp")
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="mlp",
        pointcloud_encoder_cfg=None,
        # sugar encoder params (used when encoder_type="sugar")
        sugar_encoder_cfg: dict = None,
        # flow matching params
        Conditional_ConsistencyFM=None,
        eta=0.01,
        **kwargs,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.encoder_type = encoder_type

        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # ── encoder ──────────────────────────────────────────────────────────
        if encoder_type == "sugar":
            cprint('[SugarFlowPolicy] Using SUGAR point cloud encoder', 'green')
            obs_encoder = MP1SugarEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                sugar_encoder_cfg=sugar_encoder_cfg or {},
            )
        else:
            cprint('[SugarFlowPolicy] Using MLP (PointNet) encoder', 'yellow')
            self.use_pc_color = use_pc_color
            obs_encoder = MP1Encoder(
                observation_space=obs_dict,
                img_crop_shape=crop_shape,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
            )
        # ─────────────────────────────────────────────────────────────────────

        obs_feature_dim = obs_encoder.output_shape()
        self.per_token_output = getattr(obs_encoder, 'per_token_output', False)
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim  # token_dim when per_token, else flat feat dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if Conditional_ConsistencyFM is None:
            Conditional_ConsistencyFM = {
                'eps': 1e-2, 'num_segments': 2, 'boundary': 1,
                'delta': 1e-2, 'alpha': 1e-5, 'num_inference_step': 1,
            }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        print_params(self)

    # ── inference ────────────────────────────────────────────────────────────
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        if self.encoder_type == "mlp" and not getattr(self, 'use_pc_color', False):
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                if self.per_token_output:
                    # (B*To, G+1, D) → (B, To*(G+1), D)
                    global_cond = nobs_features.reshape(B, -1, nobs_features.shape[-1])
                else:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        noise = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        z = noise.detach().clone()

        sde = ConsistencyFM('gaussian', noise_scale=1.0, use_ode_sampler='rk45',
                             sigma_var=0.0, ode_tol=1e-5, sample_N=self.num_inference_step)
        dt = 1. / sde.sample_N
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i / sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t * 99, local_cond=local_cond, global_cond=global_cond)
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = (pred
                          + (sigma_t ** 2) / (2 * (sde.noise_scale ** 2) * ((1. - num_t) ** 2))
                          * (0.5 * num_t * (1. - num_t) * pred - 0.5 * (2. - num_t) * z.detach().clone()))
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)

        z[cond_mask] = cond_data[cond_mask]
        naction_pred = z[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}

    # ── training ─────────────────────────────────────────────────────────────
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta = self.delta
        alpha = self.alpha
        reduce_op = torch.mean

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        if self.encoder_type == "mlp" and not getattr(self, 'use_pc_color', False):
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                if self.per_token_output:
                    # (B*To, G+1, D) → (B, To*(G+1), D)
                    global_cond = nobs_features.reshape(batch_size, -1, nobs_features.shape[-1])
                else:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps
        r = torch.clamp(t + delta, max=1.0)
        t_exp = t.view(-1, 1, 1).expand_as(target)
        r_exp = r.view(-1, 1, 1).expand_as(target)
        xt = t_exp * target + (1. - t_exp) * a0
        xr = r_exp * target + (1. - r_exp) * a0
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_idx = torch.searchsorted(segments, t, side="left").clamp(min=1)
        seg_ends = segments[seg_idx]
        seg_ends_exp = seg_ends.view(-1, 1, 1).expand_as(target)
        x_at_seg_ends = seg_ends_exp * target + (1. - seg_ends_exp) * a0

        def f_euler(t_e, se_e, xt_, vt_):
            return xt_ + (se_e - t_e) * vt_

        def threshold_f_euler(t_e, se_e, xt_, vt_, threshold, x_seg):
            if isinstance(threshold, int) and threshold == 0:
                return x_seg
            mask = t_e < threshold
            return mask * f_euler(t_e, se_e, xt_, vt_) + (~mask) * x_seg

        vt = self.model(xt, t * 99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r * 99, local_cond=local_cond, global_cond=global_cond)
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]
        vr = torch.nan_to_num(vr)

        ft = f_euler(t_exp, seg_ends_exp, xt, vt)
        fr = threshold_f_euler(r_exp, seg_ends_exp, xr, vr, boundary, x_at_seg_ends)

        losses_f = reduce_op(torch.square(ft - fr).reshape(batch_size, -1), dim=-1)

        def masked_losses_v(vt_, vr_, threshold, seg_ends_, t_):
            if isinstance(threshold, int) and threshold == 0:
                return 0
            lt = t_exp < threshold
            far = (seg_ends_ - t_) > 1.01 * delta
            far_exp = far.view(-1, 1, 1).expand_as(trajectory)
            lv = lt * far_exp * torch.square(vt_ - vr_)
            return reduce_op(lv.reshape(batch_size, -1), dim=-1)

        losses_v = masked_losses_v(vt, vr, boundary, seg_ends, t)
        loss = torch.mean(losses_f + alpha * losses_v)
        return loss, {'bc_loss': loss.item()}
