import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import EditingStandardDatasetBase, EditingImageConditionedMixin
from ...modules.sparse.basic import SparseTensor
from ... import models
from ...utils.render_utils import get_renderer
from ...utils.data_utils import load_balanced_group_indices
from ..structured_latent import SLatVisMixin
from tqdm import tqdm
import random


class EditingSLatVisMixin(SLatVisMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        if isinstance(x_0, SparseTensor):
            return super().visualize_sample(x_0)
        else:
            stack_images_glb1 = super().visualize_sample(x_0['edited_ss_latent'])
            stack_images_merged_glb = super().visualize_sample(x_0['ori_ss_latent'])
            # return torch.cat([stack_images_glb1, stack_images_merged_glb], dim=0)
            return torch.stack((stack_images_glb1, stack_images_merged_glb), dim=1).view(-1, *stack_images_glb1.shape[1:])  # [a[0], b[0], a[1], b[1], ...]
    
    
class EditingSLat(EditingSLatVisMixin, EditingStandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
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
        # adapt_simple_edit_data: bool = False,
        alpaca_confidence_json_path: Optional[str] = None,
        flux_edit_confidence_json_path: Optional[str] = None,
        filter_info: Optional[dict] = None,
        mixamo_root_path: Optional[str] = None,
        flux_edit_root_path: Optional[str] = None,
        alpaca_root_path: Optional[str] = None,
        dataset_info_path: Optional[str] = None,
        load_data_type: str = 'ss_latents',
        # voxel_num_json_path: Optional[str] = None,
        coords_aug_size: Optional[int] = None,
        feats_aug_grid_size: Optional[list[int]] = None,
        feats_aug_ratio: Optional[list[float]] = None,
        flux_edit_alpaca_test_json_path: Optional[str] = None,
        mixamo_test_animation: Optional[List[str]] = None,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        # augmentation for coords and feats
        self.coords_aug_size = coords_aug_size
        self.feats_aug_grid_size = feats_aug_grid_size
        self.feats_aug_ratio = feats_aug_ratio
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
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
            # adapt_simple_edit_data=adapt_simple_edit_data,
            alpaca_confidence_json_path=alpaca_confidence_json_path,
            flux_edit_confidence_json_path=flux_edit_confidence_json_path,
            filter_info=filter_info,
            mixamo_root_path=mixamo_root_path,
            flux_edit_root_path=flux_edit_root_path,
            alpaca_root_path=alpaca_root_path,
            dataset_info_path=dataset_info_path,
            load_data_type=load_data_type,
            # voxel_num_json_path=voxel_num_json_path,
            flux_edit_alpaca_test_json_path=flux_edit_alpaca_test_json_path,
            mixamo_test_animation=mixamo_test_animation,
        )

        # self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        
        # get self.loads to balance load
        self.loads = self.voxel_loads
        assert len(self.loads) == self.__len__()
        '''for index in tqdm(range(self.__len__()), desc='Calculating loads'):
            if index < len(self.mixamo_instances):
                data = self.mixamo_instances[index]
            elif index < len(self.mixamo_instances) + len(self.flux_editing_instances):
                data = self.flux_editing_instances[index - len(self.mixamo_instances)]
            else:
                data = self.alpaca_editing_instances[index - len(self.mixamo_instances) - len(self.flux_editing_instances)]
            edit_latents_path = data[1]
            edit_latents_num = self.voxel_num_json[edit_latents_path]
            self.loads.append(edit_latents_num)'''

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    # def filter_metadata(self, metadata):
    #     stats = {}
    #     metadata = metadata[metadata[f'latent_{self.latent_model}']]
    #     stats['With latent'] = len(metadata)
    #     metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
    #     stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
    #     metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
    #     stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
    #     return metadata, stats
    
    def _get_latent(self, ss_latent_path):
        data = np.load(ss_latent_path)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        # norm only on Mixamo data
        if self.normalization is not None and '3d_editing_data' not in ss_latent_path:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
        }

    def get_instance(self, ori_ss_latent_path, edited_ss_latent_path, ori_img_path, edited_img_path):
        pack = {
            'edited_ss_latent': self._get_latent(edited_ss_latent_path),
            'ori_ss_latent': self._get_latent(ori_ss_latent_path),
        }
        # augmentation only on Flux_Step_3d data
        if '3d_editing_data' in ori_ss_latent_path:
            assert '3d_editing_data' in edited_ss_latent_path
            if self.feats_aug_grid_size is not None:
                noise_ratio = random.uniform(self.feats_aug_ratio[0], self.feats_aug_ratio[1])
                feats_aug_grid_size = random.choice(self.feats_aug_grid_size)
                noise = torch.randn(1, 8, feats_aug_grid_size, feats_aug_grid_size, feats_aug_grid_size)
                noise = torch.nn.functional.interpolate(noise, size=(64, 64, 64), mode='trilinear')
                # aug on ori_ss_latent
                x_idx, y_idx, z_idx = pack['ori_ss_latent']['coords'][:, 0], pack['ori_ss_latent']['coords'][:, 1], pack['ori_ss_latent']['coords'][:, 2]
                pack['ori_ss_latent']['feats'] += noise[0, :, x_idx, y_idx, z_idx].permute(1, 0) * noise_ratio
                # aug on edited_ss_latent
                x_idx, y_idx, z_idx = pack['edited_ss_latent']['coords'][:, 0], pack['edited_ss_latent']['coords'][:, 1], pack['edited_ss_latent']['coords'][:, 2]
                pack['edited_ss_latent']['feats'] += noise[0, :, x_idx, y_idx, z_idx].permute(1, 0) * noise_ratio
            if self.coords_aug_size is not None:
                grid_aug_tensor = torch.randint(-self.coords_aug_size, self.coords_aug_size + 1, (1, 3))
                pack['edited_ss_latent']['coords'] += grid_aug_tensor
                pack['edited_ss_latent']['coords'] = torch.clamp(pack['edited_ss_latent']['coords'], min=0, max=63)
        return pack


    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['edited_ss_latent']['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            # collate ori_ss_latent
            ori_coords = []
            ori_feats = []
            ori_layout = []
            ori_start = 0 
            for i, b in enumerate(sub_batch):
                ori_coords.append(torch.cat([torch.full((b['ori_ss_latent']['coords'].shape[0], 1), i, dtype=torch.int32), b['ori_ss_latent']['coords']], dim=-1))
                ori_feats.append(b['ori_ss_latent']['feats'])
                ori_layout.append(slice(ori_start, ori_start + b['ori_ss_latent']['coords'].shape[0]))
                ori_start += b['ori_ss_latent']['coords'].shape[0]
            ori_coords = torch.cat(ori_coords)
            ori_feats = torch.cat(ori_feats)
            pack['ori_ss_latent'] = SparseTensor(
                coords=ori_coords,
                feats=ori_feats,
            )
            pack['ori_ss_latent']._shape = torch.Size([len(group), *sub_batch[0]['ori_ss_latent']['feats'].shape[1:]])
            pack['ori_ss_latent'].register_spatial_cache('layout', ori_layout)

            # collate edited_ss_latent
            edited_coords = []
            edited_feats = []
            edited_layout = []
            edited_start = 0
            for i, b in enumerate(sub_batch):
                edited_coords.append(torch.cat([torch.full((b['edited_ss_latent']['coords'].shape[0], 1), i, dtype=torch.int32), b['edited_ss_latent']['coords']], dim=-1))
                edited_feats.append(b['edited_ss_latent']['feats'])
                edited_layout.append(slice(edited_start, edited_start + b['edited_ss_latent']['coords'].shape[0]))
                edited_start += b['edited_ss_latent']['coords'].shape[0]
            edited_coords = torch.cat(edited_coords)
            edited_feats = torch.cat(edited_feats)
            pack['edited_ss_latent'] = SparseTensor(
                coords=edited_coords,
                feats=edited_feats,
            )
            pack['edited_ss_latent']._shape = torch.Size([len(group), *sub_batch[0]['edited_ss_latent']['feats'].shape[1:]])
            pack['edited_ss_latent'].register_spatial_cache('layout', edited_layout)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['ori_ss_latent', 'edited_ss_latent']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs


class EditingImageConditionedSLat(EditingImageConditionedMixin, EditingSLat):
    """
    Image conditioned structured latent dataset
    """
    pass
