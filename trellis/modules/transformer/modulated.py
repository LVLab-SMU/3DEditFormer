from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class EditingModulatedTransformerCrossBlockV0(ModulatedTransformerCrossBlock):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__(
            channels,
            ctx_channels,
            num_heads,
            mlp_ratio,
            attn_mode,
            window_size,
            shift_window,
            use_checkpoint,
            use_rope,
            qk_rms_norm,
            qk_rms_norm_cross,
            qkv_bias,
            share_mod,
        )

        self.norm_editing = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.cross_attn_editing = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )

    @torch.no_grad()
    def _init_editing_weights(self):
        # copy self.cross_attn params to self.cross_attn_editing
        self.cross_attn_editing.load_state_dict(self.cross_attn.state_dict())
        self.norm_editing.load_state_dict(self.norm2.state_dict())

        # set zeros to ['to_out.weight', 'to_out.bias']
        self.cross_attn_editing.to_out.weight.data.zero_()
        self.cross_attn_editing.to_out.bias.data.zero_()

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, context_3d: torch.Tensor, **kwargs):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        # xrh: add 3d cross-attention
        h = self.norm_editing(x)
        h = self.cross_attn_editing(h, context_3d)
        x = x + h
        # xrh: end
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, context_3d: torch.Tensor, **kwargs):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, context_3d, **kwargs, use_reentrant=False)
        else:
            return self._forward(x, mod, context, context_3d, **kwargs)
        

class EditingModulatedTransformerCrossBlockV1(ModulatedTransformerCrossBlock):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        w_only_fine_grained_feats: bool = False,
    ):
        super().__init__(
            channels,
            ctx_channels,
            num_heads,
            mlp_ratio,
            attn_mode,
            window_size,
            shift_window,
            use_checkpoint,
            use_rope,
            qk_rms_norm,
            qk_rms_norm_cross,
            qkv_bias,
            share_mod,
        )

        self.w_only_fine_grained_feats = w_only_fine_grained_feats

        # editing params
        self.norm_editing_1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        if not self.w_only_fine_grained_feats:
            self.norm_editing_2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)

        # editing params
        self.cross_attn_editing_1 = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if not self.w_only_fine_grained_feats:
            self.cross_attn_editing_2 = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )

    @torch.no_grad()
    def _init_editing_weights(self):
        # copy self.cross_attn params to self.cross_attn_editing_1 and self.cross_attn_editing_2
        self.cross_attn_editing_1.load_state_dict(self.cross_attn.state_dict())
        self.norm_editing_1.load_state_dict(self.norm2.state_dict())
        
        # set zeros to ['to_out.weight', 'to_out.bias']
        self.cross_attn_editing_1.to_out.weight.data.zero_()
        self.cross_attn_editing_1.to_out.bias.data.zero_()

        if not self.w_only_fine_grained_feats:
            self.cross_attn_editing_2.load_state_dict(self.cross_attn.state_dict())
            self.norm_editing_2.load_state_dict(self.norm2.state_dict())
            self.cross_attn_editing_2.to_out.weight.data.zero_()
            self.cross_attn_editing_2.to_out.bias.data.zero_()

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, cond_feats_3d_1: torch.Tensor, cond_feats_3d_2: torch.Tensor, get_feats_3d: bool = False):

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        # branch 1: original self-attention
        h1 = self.self_attn(h)
        
        if not get_feats_3d:
            # branch 2: cross-attention with cond_feats_3d_1
            cond_feats_3d_1 = self.norm_editing_1(cond_feats_3d_1)
            h2 = self.cross_attn_editing_1(h, cond_feats_3d_1)
            
            if not self.w_only_fine_grained_feats:
                # branch 3: cross-attention with cond_feats_3d_2
                cond_feats_3d_2 = self.norm_editing_2(cond_feats_3d_2)
                h3 = self.cross_attn_editing_2(h, cond_feats_3d_2)

                # simple fusion
                h = h1 + h2 + h3
            else:
                h = h1 + h2
                
        else:
            h = h1

        h = h * gate_msa.unsqueeze(1)
        x = x + h

        if get_feats_3d:
            feats_3d = x.clone()

        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        if get_feats_3d:
            return x, feats_3d
        else:
            return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, cond_feats_3d_1: torch.Tensor, cond_feats_3d_2: torch.Tensor, get_feats_3d: bool = False):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d, use_reentrant=False)
        else:
            return self._forward(x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d)


class EditingModulatedTransformerCrossBlockV2(EditingModulatedTransformerCrossBlockV1):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        w_only_fine_grained_feats: bool = False,
    ):
        super().__init__(
            channels,
            ctx_channels,
            num_heads,
            mlp_ratio,
            attn_mode,
            window_size,
            shift_window,
            use_checkpoint,
            use_rope,
            qk_rms_norm,
            qk_rms_norm_cross,
            qkv_bias,
            share_mod,
        )
        self.adaLN_modulation_editing = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )
        # self.fusion_proj_editing = nn.Linear(channels * 3, channels, bias=True)

    @torch.no_grad()
    def _init_editing_weights(self):
        # copy self.cross_attn params to self.cross_attn_editing_1 and self.cross_attn_editing_2
        self.cross_attn_editing_1.load_state_dict(self.cross_attn.state_dict())
        self.cross_attn_editing_2.load_state_dict(self.cross_attn.state_dict())
        self.norm_editing_1.load_state_dict(self.norm2.state_dict())
        self.norm_editing_2.load_state_dict(self.norm2.state_dict())
        
        # copy msa weight and bias
        # ori_msa_size = self.adaLN_modulation[1].weight.shape[0] // 2
        # ori_msa_weight = self.adaLN_modulation[1].weight[: ori_msa_size].repeat(2, 1)
        # ori_msa_bias = self.adaLN_modulation[1].bias[: ori_msa_size].repeat(2)
        # self.adaLN_modulation_editing[1].weight.data.copy_(ori_msa_weight)
        # self.adaLN_modulation_editing[1].bias.data.copy_(ori_msa_bias)
        self.adaLN_modulation_editing[1].weight.data.zero_()
        self.adaLN_modulation_editing[1].bias.data.zero_()

        # set zeros and ones to fusion_proj_editing
        # channels = self.fusion_proj_editing.weight.shape[0]
        # self.fusion_proj_editing.weight.data.zero_()
        # self.fusion_proj_editing.bias.data.zero_()
        # self.fusion_proj_editing.weight.data[:, :channels].copy_(torch.eye(channels))

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, cond_feats_3d_1: torch.Tensor, cond_feats_3d_2: torch.Tensor, get_feats_3d: bool = False):

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # editing branch fusion weight
        # shift_msa_1, scale_msa_1, gate_msa_1, shift_msa_2, scale_msa_2, gate_msa_2 = self.adaLN_modulation_editing(mod).chunk(6, dim=1)
        gate_msa_1, gate_msa_2 = self.adaLN_modulation_editing(mod).chunk(2, dim=1)

        # branch 1: original self-attention
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h1 = self.self_attn(h)
        h1 = h1 * gate_msa.unsqueeze(1)

        if not get_feats_3d:
            # branch 2: cross-attention with cond_feats_3d_1
            cond_feats_3d_1 = self.norm_editing_1(cond_feats_3d_1)
            h2 = self.cross_attn_editing_1(h, cond_feats_3d_1) * gate_msa_1.unsqueeze(1)

            # branch 3: cross-attention with cond_feats_3d_2
            cond_feats_3d_2 = self.norm_editing_2(cond_feats_3d_2)
            h3 = self.cross_attn_editing_2(h, cond_feats_3d_2) * gate_msa_2.unsqueeze(1)

            # simple fusion
            # h = self.fusion_proj_editing(torch.cat([h1, h2, h3], dim=-1))
            h = h1 + h2 + h3
        else:
            h = h1

        x = x + h

        if get_feats_3d:
            feats_3d = x.clone()

        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        if get_feats_3d:
            return x, feats_3d
        else:
            return x
