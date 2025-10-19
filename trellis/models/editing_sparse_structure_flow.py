from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sparse_structure_flow import TimestepEmbedder
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock, EditingModulatedTransformerCrossBlockV0, EditingModulatedTransformerCrossBlockV1, EditingModulatedTransformerCrossBlockV2
from ..modules.spatial import patchify, unpatchify


class EditingSparseStructureFlowModelV0(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        ori_img_condition: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.ori_img_condition = ori_img_condition
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

            pos_embedder_editing = AbsolutePositionEmbedder(model_channels, 3)
            pos_emb_editing = pos_embedder_editing(coords)
            self.register_buffer("pos_emb_editing", pos_emb_editing)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.input_layer_editing = nn.Linear(in_channels * patch_size**3, model_channels)
            
        self.blocks = nn.ModuleList([
            EditingModulatedTransformerCrossBlockV0(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        if self.ori_img_condition:
            self.img_condition_fusion_layer_editing = nn.Linear(cond_channels * 2, cond_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        cond_3d = patchify(kwargs['ori_ss_latent'], self.patch_size)
        cond_3d = cond_3d.view(*cond_3d.shape[:2], -1).permute(0, 2, 1).contiguous()
        cond_3d = self.input_layer_editing(cond_3d)
        cond_3d = cond_3d + self.pos_emb_editing[None]
        cond_3d = cond_3d.type(self.dtype)

        if self.ori_img_condition:
            ori_img_cond = kwargs['ori_cond_img']
            cond = torch.cat([cond, ori_img_cond], dim=-1)
            cond = self.img_condition_fusion_layer_editing(cond)

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # [1, 8, 16, 16, 16] --> [1, 4096, 8]
        h = self.input_layer(h)  # [1, 4096, 1024]
        h = h + self.pos_emb[None]  # [1, 4096, 1024]
        t_emb = self.t_embedder(t)  # [1, 1024]
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, cond_3d, **kwargs)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])  # [1, 4096, 1024]
        h = self.out_layer(h)  # [1, 4096, 1024]

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h



class EditingSparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        ori_img_condition: bool = False,
        attention_block_type_str: str = "EditingModulatedTransformerCrossBlockV0",
        ori_ss_latents_weights: float = 0.0,
        feats_3d_t: List[float] = [0.1, 0.9],
        replace_ori_ss_latents_with_noise: bool = False,
        w_only_fine_grained_feats: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.ori_img_condition = ori_img_condition
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.ori_ss_latents_weights = ori_ss_latents_weights
        self.feats_3d_t = feats_3d_t
        self.replace_ori_ss_latents_with_noise = replace_ori_ss_latents_with_noise
        self.w_only_fine_grained_feats = w_only_fine_grained_feats
        if self.w_only_fine_grained_feats:
            assert attention_block_type_str == "EditingModulatedTransformerCrossBlockV1", "w_only_fine_grained_feats is only supported for EditingModulatedTransformerCrossBlockV1"
            
        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

            # if attention_block_type_str == 'EditingModulatedTransformerCrossBlockV0':
            pos_embedder_editing = AbsolutePositionEmbedder(model_channels, 3)
            pos_emb_editing = pos_embedder_editing(coords)
            self.register_buffer("pos_emb_editing", pos_emb_editing)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        
        if attention_block_type_str == 'EditingModulatedTransformerCrossBlockV0':
            self.input_layer_editing = nn.Linear(in_channels * patch_size**3, model_channels)
        
        self.attention_block_type_str = attention_block_type_str
        attention_block_type = globals()[attention_block_type_str]
        self.blocks = nn.ModuleList([
            attention_block_type(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                w_only_fine_grained_feats=self.w_only_fine_grained_feats,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        # if self.ori_img_condition:
        #     self.img_condition_fusion_layer_editing = nn.Linear(cond_channels * 2, cond_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _init_editing_weights(self):
        for block in self.blocks:
            block._init_editing_weights()

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor,  # cond_2d_img
        get_feats_3d: bool = False, 
        cond_feats_3d_1: torch.Tensor = None, 
        cond_feats_3d_2: torch.Tensor = None, 
        **kwargs
    ) -> torch.Tensor:
        
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        if get_feats_3d:
            feats_3d_list = []

        # simple cross-attn
        if self.attention_block_type_str == 'EditingModulatedTransformerCrossBlockV0':
            cond_3d = patchify(kwargs['ori_ss_latent'], self.patch_size)
            cond_3d = cond_3d.view(*cond_3d.shape[:2], -1).permute(0, 2, 1).contiguous()
            cond_3d = self.input_layer_editing(cond_3d)
            cond_3d = cond_3d + self.pos_emb_editing[None]
            cond_3d = cond_3d.type(self.dtype)

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # [1, 8, 16, 16, 16] --> [1, 4096, 8]
        h = self.input_layer(h)  # [1, 4096, 1024]
        h = h + self.pos_emb[None]  # [1, 4096, 1024]
        t_emb = self.t_embedder(t)  # [1, 1024]
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for i, block in enumerate(self.blocks):
            if self.attention_block_type_str == 'EditingModulatedTransformerCrossBlockV0':
                h = block(h, t_emb, cond, cond_3d, **kwargs)
            elif get_feats_3d:
                h, feats_3d = block(h, t_emb, cond, cond_feats_3d_1=None, cond_feats_3d_2=None, get_feats_3d=True)  # kwargs not used
                feats_3d_list.append(feats_3d)
            else:
                h = block(h, t_emb, cond, cond_feats_3d_1[i], cond_feats_3d_2[i])  # kwargs not used
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])  # [1, 4096, 1024]
        h = self.out_layer(h)  # [1, 4096, 1024]

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        if get_feats_3d:
            return h, feats_3d_list
        else:
            return h
