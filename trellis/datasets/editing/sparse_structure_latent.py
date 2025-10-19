import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from .components import EditingStandardDatasetBase, EditingImageConditionedMixin
from ..sparse_structure_latent import SparseStructureLatentVisMixin
import random


class EditingSparseStructureLatentVisMixin(SparseStructureLatentVisMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def visualize_sample(self, x_0: Union[torch.Tensor, dict], random_offset: bool = True):
        if isinstance(x_0, torch.Tensor):
            return super().visualize_sample(x_0, random_offset)
        else:
            stack_images_glb1 = super().visualize_sample(x_0['edited_ss_latent'], random_offset)
            stack_images_merged_glb = super().visualize_sample(x_0['ori_ss_latent'], random_offset)
            # return torch.cat([stack_images_glb1, stack_images_merged_glb], dim=0)
            return torch.stack((stack_images_glb1, stack_images_merged_glb), dim=1).view(-1, *stack_images_glb1.shape[1:])  # [a[0], b[0], a[1], b[1], ...]


class EditingSparseStructureLatent(EditingSparseStructureLatentVisMixin, EditingStandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        json_file: Optional[str] = None,
        dataset_num: Optional[int] = None,
        random_cond_gt: bool = False,
        mixamo_data_repeat_ratio: float = 2.0,
        # mixamo_data_path: Optional[str] = None,
        # flux_editing_image_path: Optional[str] = None,
        # flux_editing_edited_image_path: Optional[str] = None,
        # flux_editing_3d_path: Optional[str] = None,
        # alpaca_editing_image_path: Optional[str] = None,
        # alpaca_editing_3d_path: Optional[str] = None,
        # flux_editing_confidence_json_path: Optional[str] = None,
        # alpaca_editing_confidence_json_path: Optional[str] = None,
        # filter_info: Optional[dict] = None,
        adapt_simple_edit_data: bool = False,
        alpaca_confidence_json_path: Optional[str] = None,
        flux_edit_confidence_json_path: Optional[str] = None,
        filter_info: Optional[dict] = None,
        mixamo_root_path: Optional[str] = None,
        flux_edit_root_path: Optional[str] = None,
        alpaca_root_path: Optional[str] = None,
        dataset_info_path: Optional[str] = None,
        load_data_type: str = 'ss_latents',
        voxel_aug_ratio: Optional[float] = None,
        random_ori_edit: Optional[float] = None,
        flux_edit_alpaca_test_json_path: Optional[str] = None,
        mixamo_test_animation: Optional[List[str]] = None,
        simple_edit_data_if_filtered: bool = False,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        self.voxel_aug_ratio = voxel_aug_ratio
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            json_file=json_file,
            dataset_num=dataset_num,
            random_cond_gt=random_cond_gt,
            mixamo_data_repeat_ratio=mixamo_data_repeat_ratio,
            # mixamo_data_path=mixamo_data_path,
            # flux_editing_image_path=flux_editing_image_path,
            # flux_editing_edited_image_path=flux_editing_edited_image_path,
            # flux_editing_3d_path=flux_editing_3d_path,
            # alpaca_editing_image_path=alpaca_editing_image_path,
            # alpaca_editing_3d_path=alpaca_editing_3d_path,
            # flux_editing_confidence_json_path=flux_editing_confidence_json_path,
            # alpaca_editing_confidence_json_path=alpaca_editing_confidence_json_path,
            # filter_info=filter_info,
            adapt_simple_edit_data=adapt_simple_edit_data,
            mixamo_root_path=mixamo_root_path,
            flux_edit_root_path=flux_edit_root_path,
            alpaca_root_path=alpaca_root_path,
            dataset_info_path=dataset_info_path,
            load_data_type=load_data_type,
            alpaca_confidence_json_path=alpaca_confidence_json_path,
            flux_edit_confidence_json_path=flux_edit_confidence_json_path,
            filter_info=filter_info,
            random_ori_edit=random_ori_edit,
            flux_edit_alpaca_test_json_path=flux_edit_alpaca_test_json_path,
            mixamo_test_animation=mixamo_test_animation,
            simple_edit_data_if_filtered=simple_edit_data_if_filtered,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    # def filter_metadata(self, metadata):
    #     stats = {}
    #     metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
    #     stats['With sparse structure latents'] = len(metadata)
    #     metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
    #     stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
    #     return metadata, stats

    def _get_latent(self, ss_latent_path):
        latent = np.load(ss_latent_path)
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        return z
    
    # v2 version
    def get_instance(self, ori_ss_latent_path, edited_ss_latent_path, ori_img_path, edited_img_path):
        pack = {
            'edited_ss_latent': self._get_latent(edited_ss_latent_path),
            'ori_ss_latent': self._get_latent(ori_ss_latent_path),
        }
        # augmentation only on Flux_Step_3d data and Alpaca data
        if '3d_editing_data' in ori_ss_latent_path and '3d_editing_data' in edited_ss_latent_path:
            # assert '3d_editing_data' in edited_ss_latent_path
            if self.voxel_aug_ratio is not None:
                noise_ratio = random.uniform(self.voxel_aug_ratio[0], self.voxel_aug_ratio[1])
                noise = torch.randn(8, 16, 16, 16)
                std_ratio = torch.std(pack['ori_ss_latent'], dim=0, keepdim=True)
                pack['ori_ss_latent'] += noise * noise_ratio * std_ratio
                pack['edited_ss_latent'] += noise * noise_ratio * std_ratio
        return pack

    # v1 version
    '''def _get_latent(self, root, sha256):
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{sha256}.npz'))
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std
        return z

    def get_instance(self, root, glb1_sha256, merged_glb_sha256):
        pack = {
            'edited_ss_latent': self._get_latent(root, glb1_sha256),
            'ori_ss_latent': self._get_latent(root, merged_glb_sha256),
        }
        return pack'''
    

class EditingImageConditionedSparseStructureLatent(EditingImageConditionedMixin, EditingSparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass
    