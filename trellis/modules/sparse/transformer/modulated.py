from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet
from copy import deepcopy


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class EditingModulatedSparseTransformerCrossBlockV0(ModulatedSparseTransformerCrossBlock):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
            shift_sequence,
            shift_window,
            serialize_mode,
            use_checkpoint,
            use_rope,
            qk_rms_norm,
            qk_rms_norm_cross,
            qkv_bias,
            share_mod,
        )

        # editing params
        self.norm_editing = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)

        # editing params
        self.cross_attn_editing = SparseMultiHeadAttention(
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
        
    def _forward(
        self, 
        x: SparseTensor, 
        mod: torch.Tensor, 
        context: torch.Tensor, 
        context_3d: SparseTensor,
    ) -> SparseTensor:

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # editing branch fusion weight
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        # xrh: add 3d cross-attention
        h = x.replace(self.norm_editing(x.feats))
        h = self.cross_attn_editing(h, context_3d)
        x = x + h
        # xrh: end
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, context_3d: SparseTensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, context_3d, use_reentrant=False)
        else:
            return self._forward(x, mod, context, context_3d)


class EditingModulatedSparseTransformerCrossBlockV1(ModulatedSparseTransformerCrossBlock):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
            shift_sequence,
            shift_window,
            serialize_mode,
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
        self.cross_attn_editing_1 = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if not self.w_only_fine_grained_feats:
            self.cross_attn_editing_2 = SparseMultiHeadAttention(
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
        
    def _forward(
        self, 
        x: SparseTensor, 
        mod: torch.Tensor, 
        context: torch.Tensor, 
        cond_feats_3d_1: SparseTensor, 
        cond_feats_3d_2: SparseTensor, 
        get_feats_3d: bool = False
    ) -> SparseTensor:

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # editing branch fusion weight
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h1 = self.self_attn(h)
        h1 = h1 * gate_msa

        if not get_feats_3d:
            # branch 2: cross-attention with cond_feats_3d_1
            cond_feats_3d_1_norm = cond_feats_3d_1.replace(self.norm_editing_1(cond_feats_3d_1.feats))
            h2 = self.cross_attn_editing_1(h, cond_feats_3d_1_norm)

            if not self.w_only_fine_grained_feats:
                # branch 3: cross-attention with cond_feats_3d_2
                cond_feats_3d_2_norm = cond_feats_3d_2.replace(self.norm_editing_2(cond_feats_3d_2.feats))
                h3 = self.cross_attn_editing_2(h, cond_feats_3d_2_norm)

                # simple fusion
                h = h1 + h2 + h3
            else:
                h = h1 + h2
                
        else:
            h = h1

        x = x + h

        if get_feats_3d:
            feats_3d = deepcopy(x)

        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        if get_feats_3d:
            return x, feats_3d
        else:
            return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, cond_feats_3d_1: SparseTensor, cond_feats_3d_2: SparseTensor, get_feats_3d: bool = False) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d, use_reentrant=False)
        else:
            return self._forward(x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d)


class EditingModulatedSparseTransformerCrossBlockV2(ModulatedSparseTransformerCrossBlock):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
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
            shift_sequence,
            shift_window,
            serialize_mode,
            use_checkpoint,
            use_rope,
            qk_rms_norm,
            qk_rms_norm_cross,
            qkv_bias,
            share_mod,
        )

        # editing params
        self.norm_editing_1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm_editing_2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)

        # editing params
        self.cross_attn_editing_1 = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.cross_attn_editing_2 = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )

        self.adaLN_modulation_editing = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )

    @torch.no_grad()
    def _init_editing_weights(self):
        # copy self.cross_attn params to self.cross_attn_editing_1 and self.cross_attn_editing_2
        self.cross_attn_editing_1.load_state_dict(self.cross_attn.state_dict())
        self.cross_attn_editing_2.load_state_dict(self.cross_attn.state_dict())
        self.norm_editing_1.load_state_dict(self.norm2.state_dict())
        self.norm_editing_2.load_state_dict(self.norm2.state_dict())

        # set zeros to ['to_out.weight', 'to_out.bias']
        # self.cross_attn_editing_1.to_out.weight.data.zero_()
        # self.cross_attn_editing_1.to_out.bias.data.zero_()
        # self.cross_attn_editing_2.to_out.weight.data.zero_()
        # self.cross_attn_editing_2.to_out.bias.data.zero_()

        self.adaLN_modulation_editing[1].weight.data.zero_()
        self.adaLN_modulation_editing[1].bias.data.zero_()
        
    def _forward(
        self, 
        x: SparseTensor, 
        mod: torch.Tensor, 
        context: torch.Tensor, 
        cond_feats_3d_1: SparseTensor, 
        cond_feats_3d_2: SparseTensor, 
        get_feats_3d: bool = False
    ) -> SparseTensor:

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        # editing branch fusion weight
        gate_msa_1, gate_msa_2 = self.adaLN_modulation_editing(mod).chunk(2, dim=1)

        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h1 = self.self_attn(h)
        h1 = h1 * gate_msa

        if not get_feats_3d:
            # branch 2: cross-attention with cond_feats_3d_1
            cond_feats_3d_1_norm = cond_feats_3d_1.replace(self.norm_editing_1(cond_feats_3d_1.feats))
            h2 = self.cross_attn_editing_1(h, cond_feats_3d_1_norm) * gate_msa_1#[h.coords[:, 0]]

            # branch 3: cross-attention with cond_feats_3d_2
            cond_feats_3d_2_norm = cond_feats_3d_2.replace(self.norm_editing_2(cond_feats_3d_2.feats))
            h3 = self.cross_attn_editing_2(h, cond_feats_3d_2_norm) * gate_msa_2#[h.coords[:, 0]]

            # simple fusion
            h = h1 + h2 + h3
        else:
            h = h1

        x = x + h

        if get_feats_3d:
            feats_3d = deepcopy(x)

        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        if get_feats_3d:
            return x, feats_3d
        else:
            return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor, cond_feats_3d_1: SparseTensor, cond_feats_3d_2: SparseTensor, get_feats_3d: bool = False) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d, use_reentrant=False)
        else:
            return self._forward(x, mod, context, cond_feats_3d_1, cond_feats_3d_2, get_feats_3d)
