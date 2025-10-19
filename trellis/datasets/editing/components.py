from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json
from collections import defaultdict
import random
import math
import rembg
import copy


class EditingStandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
        json_file: str,
        dataset_num: Optional[int] = None,
        random_cond_gt: bool = False,
        # mixamo_data_path: Optional[str] = None,
        mixamo_data_repeat_ratio: float = 2.0,
        # flux_editing_image_path: Optional[str] = None,
        # flux_editing_edited_image_path: Optional[str] = None,
        # flux_editing_3d_path: Optional[str] = None,
        # alpaca_editing_image_path: Optional[str] = None,
        # alpaca_editing_3d_path: Optional[str] = None,
        # flux_editing_confidence_json_path: Optional[str] = None,
        # alpaca_editing_confidence_json_path: Optional[str] = None,
        # filter_info: Optional[dict] = None,
        adapt_simple_edit_data: bool = False,
        load_data_type: str = 'ss_latents',
        # voxel_num_json_path: Optional[str] = None,
        mixamo_root_path: Optional[str] = None,
        flux_edit_root_path: Optional[str] = None,
        alpaca_root_path: Optional[str] = None,
        dataset_info_path: Optional[str] = None,
        alpaca_confidence_json_path: Optional[str] = None,
        flux_edit_confidence_json_path: Optional[str] = None,
        filter_info: Optional[dict] = None,
        random_ori_edit: Optional[float] = None,
        flux_edit_alpaca_test_json_path: Optional[str] = None,
        mixamo_test_animation: Optional[List[str]] = None,
        simple_edit_data_if_filtered: bool = False,
    ):
        super().__init__()
        self.roots = roots
        self.json_file = json_file
        self.dataset_num = dataset_num
        self.random_cond_gt = random_cond_gt
        # self.mixamo_data_path = mixamo_data_path
        self.mixamo_data_repeat_ratio = mixamo_data_repeat_ratio
        # self.flux_editing_image_path = flux_editing_image_path
        # self.flux_editing_edited_image_path = flux_editing_edited_image_path
        # self.flux_editing_3d_path = flux_editing_3d_path
        # self.alpaca_editing_image_path = alpaca_editing_image_path
        # self.alpaca_editing_3d_path = alpaca_editing_3d_path
        # self.flux_editing_confidence_json_path = flux_editing_confidence_json_path
        # self.alpaca_editing_confidence_json_path = alpaca_editing_confidence_json_path
        # self.filter_info = filter_info
        self.adapt_simple_edit_data = adapt_simple_edit_data
        self.load_data_type = load_data_type
        # self.voxel_num_json_path = voxel_num_json_path
        self.mixamo_root_path = mixamo_root_path
        self.flux_edit_root_path = flux_edit_root_path
        self.alpaca_root_path = alpaca_root_path
        self.dataset_info_path = dataset_info_path
        self.alpaca_confidence_json_path = alpaca_confidence_json_path
        self.flux_edit_confidence_json_path = flux_edit_confidence_json_path
        self.filter_info = filter_info
        self.random_ori_edit = random_ori_edit
        self.flux_edit_alpaca_test_json_path = flux_edit_alpaca_test_json_path
        self.mixamo_test_animation = mixamo_test_animation
        self.simple_edit_data_if_filtered = simple_edit_data_if_filtered
        assert self.load_data_type in ['ss_latents', 'latents'], f'Invalid load_data_type: {self.load_data_type}'
        if self.filter_info is not None:
            with open(self.flux_edit_confidence_json_path, 'r') as f:
                self.flux_edit_confidence_json = json.load(f)
            with open(self.alpaca_confidence_json_path, 'r') as f:
                self.alpaca_confidence_json = json.load(f)
        with open(self.dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        if self.random_ori_edit is not None:
            assert self.load_data_type == 'ss_latents', f'Invalid load_data_type: {self.load_data_type}'
        if self.flux_edit_alpaca_test_json_path is not None:
            with open(self.flux_edit_alpaca_test_json_path, 'r') as f:
                self.flux_edit_alpaca_test_json = json.load(f)
            self.flux_edit_alpaca_test_json = self.flux_edit_alpaca_test_json['alpaca'] + self.flux_edit_alpaca_test_json['flux_edit']
            assert isinstance(self.flux_edit_alpaca_test_json, list)
            test_num = len(self.flux_edit_alpaca_test_json)
            self.flux_edit_alpaca_test_json = set(self.flux_edit_alpaca_test_json)
            assert test_num == len(self.flux_edit_alpaca_test_json)

        # v3 version
        # ---- prepare voxel num
        self.voxel_loads = []
        # ---- load mixamo data
        self.mixamo_instances = self.dataset_info['mixamo']
        print('Ori mixamo data num: ', sum(len(instances) for instances in self.mixamo_instances.values()))
        _mixamo_instances = []
        for character_name, instances in self.mixamo_instances.items():
            if self.mixamo_test_animation is not None and character_name in self.mixamo_test_animation:
                continue
            object_num = 0
            for _ in range(math.ceil(self.mixamo_data_repeat_ratio)):
                random.shuffle(instances)
                i, j = 0, len(instances) - 1
                while i < j:
                    _mixamo_instances.append([
                        os.path.join(self.mixamo_root_path, instances[i]['ss_latents_path' if self.load_data_type == 'ss_latents' else 'latents_path']),
                        os.path.join(self.mixamo_root_path, instances[j]['ss_latents_path' if self.load_data_type == 'ss_latents' else 'latents_path']),
                        os.path.join(self.mixamo_root_path, instances[i]['img_path']),
                        os.path.join(self.mixamo_root_path, instances[j]['img_path']),
                    ])
                    self.voxel_loads.append((instances[i]['voxel_num'] + instances[j]['voxel_num']) // 2)
                    object_num += 1
                    i += 1
                    j -= 1
                    if object_num >= len(instances) * self.mixamo_data_repeat_ratio:
                        break
        self.mixamo_instances = _mixamo_instances
        print(f'Repeated Mixamo data num: {len(self.mixamo_instances)}')

        # ---- load flux editing data
        self.flux_editing_instances = []
        remove_num = 0
        test_num = 0
        for key, value in self.dataset_info['flux_edit'].items():

            # load data
            ori_ss_latents_path = value['ori_ss_latents_path' if self.load_data_type == 'ss_latents' else 'ori_latents_path']
            edited_ss_latents_path = value['edit_ss_latents_path' if self.load_data_type == 'ss_latents' else 'edit_latents_path']

            # remove test data
            if self.flux_edit_alpaca_test_json_path is not None and key in self.flux_edit_alpaca_test_json:
                test_num += 1
                continue

            # remove filtered data
            if self.filter_info is not None:
                choose_flag = True
                for filter_key, filter_range in filter_info.items():
                    min_val, max_val = filter_range
                    if self.flux_edit_confidence_json[key][filter_key] < min_val or self.flux_edit_confidence_json[key][filter_key] > max_val:
                        choose_flag = False
                        break
                if not choose_flag:
                    if not self.simple_edit_data_if_filtered:
                        remove_num += 1
                        continue
                    else:
                        assert self.load_data_type == 'ss_latents'
                        edited_ss_latents_path = edited_ss_latents_path.replace('edit_ss', 'simple_edit_ss').replace('data_v4', 'data_v2')
                        assert os.path.exists(os.path.join(self.flux_edit_root_path, edited_ss_latents_path)), f'{edited_ss_latents_path} not found'
            
            # load data
            if self.adapt_simple_edit_data:
                assert self.load_data_type == 'ss_latents'
                edited_ss_latents_path = edited_ss_latents_path.replace('edit_ss', 'simple_edit_ss').replace('data_v4', 'data_v2')
                assert os.path.exists(os.path.join(self.flux_edit_root_path, edited_ss_latents_path)), f'{edited_ss_latents_path} not found'
            self.flux_editing_instances.append([
                os.path.join(self.flux_edit_root_path, ori_ss_latents_path),
                os.path.join(self.flux_edit_root_path, edited_ss_latents_path),
                os.path.join(self.flux_edit_root_path, value['ori_img_path']),
                os.path.join(self.flux_edit_root_path, value['edit_img_path']),
            ])
            self.voxel_loads.append((value['ori_voxel_num'] + value['edit_voxel_num']) // 2)
        print(f'Remove flux editing data (based on confidence) num: {remove_num}')
        print(f'Remove test flux editing data num: {test_num}')
        print(f'Flux editing data num: {len(self.flux_editing_instances)}')

        # ---- load alpaca editing data
        self.alpaca_editing_instances = []
        remove_num = 0
        test_num = 0
        for key, value in self.dataset_info['alpaca'].items():

            # load data
            ori_ss_latents_path = value['ori_ss_latents_path' if self.load_data_type == 'ss_latents' else 'ori_latents_path']
            edited_ss_latents_path = value['edit_ss_latents_path' if self.load_data_type == 'ss_latents' else 'edit_latents_path']

            # remove test data
            if self.flux_edit_alpaca_test_json_path is not None and key in self.flux_edit_alpaca_test_json:
                test_num += 1
                continue

            # remove filtered data
            if self.filter_info is not None:
                choose_flag = True
                for filter_key, filter_range in filter_info.items():
                    min_val, max_val = filter_range
                    if self.alpaca_confidence_json[key][filter_key] < min_val or self.alpaca_confidence_json[key][filter_key] > max_val:
                        choose_flag = False
                        break
                if not choose_flag:
                    if not self.simple_edit_data_if_filtered:
                        remove_num += 1
                        continue
                    else:
                        assert self.load_data_type == 'ss_latents'
                        edited_ss_latents_path = edited_ss_latents_path.replace('edit_ss', 'simple_edit_ss').replace('data_V3', 'data_V2')
                        assert os.path.exists(os.path.join(self.alpaca_root_path, edited_ss_latents_path)), f'{edited_ss_latents_path} not found'
            
            # load data
            if self.adapt_simple_edit_data:
                assert self.load_data_type == 'ss_latents'
                edited_ss_latents_path = edited_ss_latents_path.replace('edit_ss', 'simple_edit_ss').replace('data_V3', 'data_V2')
                assert os.path.exists(os.path.join(self.alpaca_root_path, edited_ss_latents_path)), f'{edited_ss_latents_path} not found'
            self.alpaca_editing_instances.append([
                os.path.join(self.alpaca_root_path, ori_ss_latents_path),
                os.path.join(self.alpaca_root_path, edited_ss_latents_path),
                os.path.join(self.alpaca_root_path, value['ori_img_path']),
                os.path.join(self.alpaca_root_path, value['edit_img_path']),
            ])
            self.voxel_loads.append((value['ori_voxel_num'] + value['edit_voxel_num']) // 2)
        print(f'Remove alpaca editing data (based on confidence) num: {remove_num}')
        print(f'Remove test alpaca editing data num: {test_num}')
        print(f'Alpaca editing data num: {len(self.alpaca_editing_instances)}')

        # v2 version
        '''# ---- load mixamo data
        self.mixamo_instances = defaultdict(list)
        if self.load_data_type == 'ss_latents':
            self.mixamo_ss_latents_path = os.path.join(self.mixamo_data_path, 'ss_latents', 'ss_enc_conv3d_16l8_fp16')
        else:
            self.mixamo_ss_latents_path = os.path.join(self.mixamo_data_path, 'latents', 'dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16')
        self.mixamo_renders_cond_path = os.path.join(self.mixamo_data_path, 'renders_cond')
        self.mixamo_ss_latents_list = os.listdir(self.mixamo_ss_latents_path)
        self.mixamo_ss_latents_list.sort()
        for ss_latent_name in self.mixamo_ss_latents_list:
            character_name = ss_latent_name.split('_')[0]
            cond_img_path = os.path.join(self.mixamo_data_path, 'renders_cond', ss_latent_name[:-4], '000.png')
            assert os.path.exists(cond_img_path), f'{cond_img_path} not found'
            self.mixamo_instances[character_name].append([
                os.path.join(self.mixamo_ss_latents_path, ss_latent_name),
                os.path.join(self.mixamo_renders_cond_path, ss_latent_name[:-4], '000.png')
            ])
        print('Ori mixamo data num: ', len(self.mixamo_ss_latents_list))
        _mixamo_instances = []
        for character_name, instances in self.mixamo_instances.items():
            object_num = 0
            for _ in range(math.ceil(self.mixamo_data_repeat_ratio)):
                random.shuffle(instances)
                i, j = 0, len(instances) - 1
                while i < j:
                    _mixamo_instances.append([
                        instances[i][0],
                        instances[j][0],
                        instances[i][1],
                        instances[j][1]
                    ])
                    object_num += 1
                    i += 1
                    j -= 1
                    if object_num >= len(instances) * self.mixamo_data_repeat_ratio:
                        break
        self.mixamo_instances = _mixamo_instances
        print(f'Repeated Mixamo data num: {len(self.mixamo_instances)}')

        # ---- load flux editing data
        self.flux_editing_instances = []
        self.flux_editing_3d_list = os.listdir(self.flux_editing_3d_path)
        self.flux_editing_3d_list.sort()
        remove_num = 0
        for flux_editing_3d_name in self.flux_editing_3d_list:
            if self.load_data_type == 'ss_latents':
                ori_ss_latent_path = os.path.join(self.flux_editing_3d_path, flux_editing_3d_name, 'ori_ss_latents.npz')
                edited_ss_latent_path = os.path.join(self.flux_editing_3d_path, flux_editing_3d_name, 'edit_ss_latents.npz')
            else:
                ori_ss_latent_path = os.path.join(self.flux_editing_3d_path, flux_editing_3d_name, 'ori_latents.npz')
                edited_ss_latent_path = os.path.join(self.flux_editing_3d_path, flux_editing_3d_name, 'edit_latents.npz').replace('Flux_Step_3d_editing_data_v2', 'Flux_Step_3d_editing_data_v3')
                if not os.path.exists(edited_ss_latent_path):
                    continue
            if not (os.path.exists(ori_ss_latent_path) and os.path.exists(edited_ss_latent_path)):
                continue
            if self.adapt_simple_edit_data:
                edited_ss_latent_path = edited_ss_latent_path.replace('edit_ss_latents.npz', 'simple_edit_ss_latents.npz')
                assert os.path.exists(edited_ss_latent_path), f'{edited_ss_latent_path} not found'
            if self.filter_info is not None and self.flux_editing_confidence_json_path is not None:
                choose_flag = True
                for filter_key, filter_range in filter_info.items():
                    min_val, max_val = filter_range
                    if self.flux_editing_confidence_json[flux_editing_3d_name][filter_key] < min_val or self.flux_editing_confidence_json[flux_editing_3d_name][filter_key] > max_val:
                        choose_flag = False
                        break
                if not choose_flag:
                    remove_num += 1
                    continue
            object_name, index = flux_editing_3d_name.split('_')
            ori_img_path = os.path.join(self.flux_editing_image_path, object_name, f'ori_{index}.png')
            # edited_img_path = os.path.join(self.flux_editing_image_path, object_name, f'edited_{index}.png')
            edit_img_list = os.listdir(os.path.join(self.flux_editing_edited_image_path, flux_editing_3d_name))
            edit_img_list.sort()
            edited_img_path = os.path.join(self.flux_editing_edited_image_path, flux_editing_3d_name, edit_img_list[-1])
            assert os.path.exists(ori_img_path), f'{ori_img_path} not found'
            assert os.path.exists(edited_img_path), f'{edited_img_path} not found'
            self.flux_editing_instances.append([
                ori_ss_latent_path,
                edited_ss_latent_path,
                ori_img_path,
                edited_img_path,
            ])
        print(f'Remove flux editing data (based on confidence) num: {remove_num}')
        print(f'Flux editing data num: {len(self.flux_editing_instances)}')

        # ---- load alpaca editing data
        self.alpaca_editing_instances = []
        self.alpaca_editing_3d_list = os.listdir(self.alpaca_editing_3d_path)
        self.alpaca_editing_3d_list.sort()
        remove_num = 0
        if self.load_data_type == 'ss_latents':
            for alpaca_editing_3d_name in self.alpaca_editing_3d_list:
                ori_ss_latent_path = os.path.join(self.alpaca_editing_3d_path, alpaca_editing_3d_name, 'ori_ss_latents.npz')
                edited_ss_latent_path = os.path.join(self.alpaca_editing_3d_path, alpaca_editing_3d_name, 'edit_ss_latents.npz')
                if not (os.path.exists(ori_ss_latent_path) and os.path.exists(edited_ss_latent_path)):
                    continue
                if self.filter_info is not None and self.alpaca_editing_confidence_json_path is not None:
                    choose_flag = True
                    for filter_key, filter_range in filter_info.items():
                        min_val, max_val = filter_range
                        if self.alpaca_editing_confidence_json[alpaca_editing_3d_name][filter_key] < min_val or self.alpaca_editing_confidence_json[alpaca_editing_3d_name][filter_key] > max_val:
                            choose_flag = False
                            break
                    if not choose_flag:
                        remove_num += 1
                        continue
                ori_img_path = os.path.join(self.alpaca_editing_image_path, alpaca_editing_3d_name, 'original.png')
                if not os.path.exists(ori_img_path):
                    ori_img_path = ori_img_path.replace('open_data', 'open_data2')
                    assert os.path.exists(ori_img_path), f'{ori_img_path} not found'
                edited_img_path = ori_img_path.replace('original.png', 'after_edited_Step1X-Edit.png')
                assert os.path.exists(edited_img_path), f'{edited_img_path} not found'
                self.alpaca_editing_instances.append([
                    ori_ss_latent_path,
                    edited_ss_latent_path,
                    ori_img_path,
                    edited_img_path,
                ])
        print(f'Remove alpaca editing data (based on confidence) num: {remove_num}')
        print(f'Alpaca editing data num: {len(self.alpaca_editing_instances)}')'''

        print('**********************')
        print(f'Total editing data num: {len(self.mixamo_instances) + len(self.flux_editing_instances) + len(self.alpaca_editing_instances)}, including:')
        print(f'  - Mixamo: {len(self.mixamo_instances)}')
        print(f'  - Flux editing: {len(self.flux_editing_instances)}')
        print(f'  - Alpaca editing: {len(self.alpaca_editing_instances)}')
        print('**********************')

        # v1 version
        '''with open(json_file, 'r') as f:
            json_info = json.load(f)
        json_info = json_info[:dataset_num]
        print('Plan to load {} instances'.format(len(json_info)))

        self.instances = []
        for info in json_info:
            glb1_path = info['model1_path']
            merged_glb_path = info['merged_glb_path']
            glb1_sha256 = glb1_path.split('/')[-1].split('.')[0]
            merged_glb_sha256 = merged_glb_path.split('/')[-1].split('.')[0]
            exist_glb1 = self.exist_instance(self.roots, glb1_sha256)
            exist_merged_glb = self.exist_instance(self.roots, merged_glb_sha256)
            if exist_glb1 and exist_merged_glb:
                self.instances.append([self.roots, glb1_sha256, merged_glb_sha256])   
        print('Actually load {} instances'.format(len(self.instances)))'''
        
        # ori version
        '''self.metadata = pd.DataFrame()
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])'''
    
    def exist_instance(self, root, sha256):
        cond1 = os.path.exists(os.path.join(root, 'ss_latents', 'ss_enc_conv3d_16l8_fp16', f'{sha256}.npz'))
        cond2 = os.path.exists(os.path.join(root, 'renders_cond', sha256, 'transforms.json'))
        return cond1 and cond2
    
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    # @abstractmethod
    # def get_instance(self, root: str, glb1_sha256: str, merged_glb_sha256: str) -> Dict[str, Any]:
    #     pass
    
    @abstractmethod
    def get_instance(self, ori_ss_latent_path: str, edited_ss_latent_path: str, ori_img_path: str, edited_img_path: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        # return len(self.instances)
        return len(self.mixamo_instances) + len(self.flux_editing_instances) + len(self.alpaca_editing_instances)

    def get_data_path_from_index(self, index) -> Dict[str, Any]:
        if index < len(self.mixamo_instances):
            data = self.mixamo_instances[index]
        elif index < len(self.mixamo_instances) + len(self.flux_editing_instances):
            data = self.flux_editing_instances[index - len(self.mixamo_instances)]
        else:
            data = self.alpaca_editing_instances[index - len(self.mixamo_instances) - len(self.flux_editing_instances)]
        return data

    def __getitem__(self, index) -> Dict[str, Any]:
        # try:
        data_path = self.get_data_path_from_index(index)
        if self.random_ori_edit is not None and np.random.rand() < self.random_ori_edit:
            another_index = np.random.randint(0, len(self))
            another_data_path = self.get_data_path_from_index(another_index)
            data_path = [another_data_path[0], data_path[1], another_data_path[2], data_path[3]]
        return self.get_instance(*data_path)
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        # v2 version
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        lines.append(f'    - Mixamo: {len(self.mixamo_instances)}')
        lines.append(f'    - Flux editing: {len(self.flux_editing_instances)}')
        lines.append(f'    - Alpaca editing: {len(self.alpaca_editing_instances)}')
        return '\n'.join(lines)
    
        # v1 version
        '''lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for root, glb1_sha256, merged_glb_sha256 in self.instances:
            lines.append(f'    - {root}:')
            lines.append(f'      - {glb1_sha256}:')
            lines.append(f'      - {merged_glb_sha256}:')
        return '\n'.join(lines)'''
    
    
class EditingImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
        # lazy init rembg session
        self.rembg_session = None
    
    def __deepcopy__(self, memo):
        """
        自定义深度复制逻辑。
        """
        # print("正在调用自定义的 __deepcopy__ ...")
        
        # 1. 创建一个新的类实例，但不调用 __init__，避免重复创建 session
        cls = self.__class__
        result = cls.__new__(cls)
        
        # memo 是一个字典，用来处理循环引用，这是实现 __deepcopy__ 的标准做法
        memo[id(self)] = result

        # 2. 遍历 self 的 __dict__，深度复制所有其他属性
        for k, v in self.__dict__.items():
            if k == 'rembg_session':
                # 跳过 rembg_session 的复制
                # print(f"跳过复制 'rembg_session' 属性")
                continue
            setattr(result, k, copy.deepcopy(v, memo))

        # 3. 在新的副本中，手动创建一个全新的 rembg_session
        # print("在新的副本中创建一个全新的 rembg_session...")
        result.rembg_session = rembg.new_session('u2net')
        
        return result

    # def filter_metadata(self, metadata):
    #     metadata, stats = super().filter_metadata(metadata)
    #     metadata = metadata[metadata[f'cond_rendered']]
    #     stats['Cond rendered'] = len(metadata)
    #     return metadata, stats
    
    def _get_cond_image(self, image_path, aug_bbox=None):
        # v1 version
        '''image_root = os.path.join(root, 'renders_cond', sha256)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views) if view_idx is None else view_idx
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])'''
        image = Image.open(image_path)
        if np.array(image).shape[-1] != 4:
            if self.rembg_session is None:
                # Initializing rembg session
                self.rembg_session = rembg.new_session('u2net')
            image = rembg.remove(image, session=self.rembg_session)

        alpha = np.array(image.getchannel(3))
        if aug_bbox is None:
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_size_ratio = 1.2
            aug_hsize = hsize * aug_size_ratio
            aug_center_offset = [0, 0]
            aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
            aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        # return image, view, aug_bbox
        return image, aug_bbox

    def get_instance(self, ori_ss_latent_path, edited_ss_latent_path, ori_img_path, edited_img_path):
        pack = super().get_instance(ori_ss_latent_path, edited_ss_latent_path, ori_img_path, edited_img_path)
       
        ori_cond_img, _ = self._get_cond_image(ori_img_path)
        edited_cond_img, _ = self._get_cond_image(edited_img_path)

        pack['ori_cond_img'] = ori_cond_img
        pack['edited_cond_img'] = edited_cond_img

        if self.random_cond_gt and np.random.rand() < 0.5:
            pack['ori_cond_img'], pack['edited_cond_img'] = pack['edited_cond_img'], pack['ori_cond_img']
            pack['ori_ss_latent'], pack['edited_ss_latent'] = pack['edited_ss_latent'], pack['ori_ss_latent']

        return pack

    