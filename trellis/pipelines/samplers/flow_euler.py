from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        edit_condition: Optional[Any] = None,
        start_step: Optional[Any] = None,
        end_step: Optional[Any] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise

        # get ori_ss_latents_weights
        if hasattr(model, 'module') and hasattr(model.module, 'ori_ss_latents_weights'):
            ori_ss_latents_weights = model.module.ori_ss_latents_weights
        elif hasattr(model, 'ori_ss_latents_weights'):
            ori_ss_latents_weights = model.ori_ss_latents_weights
        else:
            ori_ss_latents_weights = None  # ori_ss_latents_weights only in ss_latents stage
        if ori_ss_latents_weights is not None and ori_ss_latents_weights != 0.0:
            sample = noise * (1 - ori_ss_latents_weights) + kwargs['ori_ss_latent'] * ori_ss_latents_weights
            
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        if start_step is not None and end_step is not None:
            t_pairs = t_pairs[start_step: end_step]
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        # get feats_3d_1 and feats_3d_2
        t = torch.tensor([1000 * t_pairs[0][0]] * noise.shape[0], device=noise.device, dtype=torch.float32)
        if 'ori_ss_latent' in kwargs and (not hasattr(model, 'input_layer_editing')):
            with torch.no_grad():
                t1, t2 = model.feats_3d_t
                _, feats_3d_1 = model(kwargs['ori_ss_latent'], (torch.zeros_like(t) + t1) * 1000, torch.zeros_like(cond), get_feats_3d=True, **kwargs)  # 24 * [B, 4096, 1024]
                _, feats_3d_2 = model(kwargs['ori_ss_latent'], (torch.zeros_like(t) + t2) * 1000, cond, get_feats_3d=True, **kwargs)  # 24 * [B, 4096, 1024]
            kwargs['cond_feats_3d_1'] = feats_3d_1
            kwargs['cond_feats_3d_2'] = feats_3d_2

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            if edit_condition is not None and edit_condition['edit_mask'] is not None:
                if isinstance(noise, torch.Tensor):
                    noised_condition = edit_condition['z_sparse_structure'] * (1 - t_prev) + noise * t_prev
                    edit_mask = torch.nn.functional.interpolate(edit_condition['edit_mask'], size=sample.shape[2:], mode='nearest')
                    out.pred_x_prev = out.pred_x_prev * edit_mask + noised_condition * (1 - edit_mask)
                else:
                    condition_feats = edit_condition['z_slat_feats']
                    condition_coords = edit_condition['z_slat_coords']
                    edit_mask_3d = edit_condition['edit_mask'].bool()
                    pred_coords_long = out.pred_x_prev.coords.long()
                    condition_coords_long = condition_coords.long()

                    # get the noised condition
                    noised_condition = condition_feats * (1 - t_prev) + torch.randn_like(condition_feats).to(noise.device) * t_prev

                    # get the mask values for the predicted coords
                    mask_values_for_pred = edit_mask_3d[
                        pred_coords_long[:, 0],
                        0,
                        pred_coords_long[:, 1],
                        pred_coords_long[:, 2],
                        pred_coords_long[:, 3]
                    ]
                    filtered_pred_coords_tensor = out.pred_x_prev.coords[mask_values_for_pred]
                    filtered_pred_feats_tensor = out.pred_x_prev.feats[mask_values_for_pred]

                    # get the mask values for the condition coords
                    mask_values_for_cond = edit_mask_3d[
                        condition_coords_long[:, 0],
                        0,
                        condition_coords_long[:, 1],
                        condition_coords_long[:, 2],
                        condition_coords_long[:, 3]
                    ]
                    filtered_cond_coords_tensor = condition_coords[~mask_values_for_cond]
                    filtered_cond_feats_tensor = noised_condition[~mask_values_for_cond]

                    # merge the predicted and condition coords
                    merged_coords = torch.cat((filtered_pred_coords_tensor, filtered_cond_coords_tensor), dim=0)
                    merged_feats = torch.cat((filtered_pred_feats_tensor, filtered_cond_feats_tensor), dim=0)
                    out.pred_x_prev = out.pred_x_prev.replace(coords=merged_coords, feats=merged_feats)
                    
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
