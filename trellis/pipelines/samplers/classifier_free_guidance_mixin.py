from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        if cfg_strength != 0:
            if 'cond_feats_3d_1' in kwargs:
                kwargs['get_feats_3d'] = True
                neg_pred, _ = super()._inference_model(model, x_t, t, cond, **kwargs)
            else:
                neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        return pred
