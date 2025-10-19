import os
import json
from easydict import EasyDict as edict
from trellis import models, datasets, trainers
import copy
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from IPython.display import display
import torch
import numpy as np
import random
from peft import LoraConfig, LoraModel
import time
from trellis.datasets.sparse_structure_latent import SparseStructureLatentVisMixin
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import rembg
import utils3d
from trellis.renderers import OctreeRenderer
from trellis.representations import Octree
import imageio
import os
from trellis.modules import sparse as sp
import argparse
from tqdm import tqdm
from subprocess import DEVNULL, call, TimeoutExpired
import math
import queue
import threading

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _render(file_path, engine='CYCLES', cuda_idx=0, total_render_view_num=300, eval_view_num=10, BLENDER_EEVEE_NEXT_on_0_device=False, timeout_seconds=600, no_norm_scene=False, debug=False, blender_path=''):

    output_folder = os.path.join(os.path.dirname(file_path), 'render')
    os.makedirs(output_folder, exist_ok=True)
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return output_folder

    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    # sphere_hammersley_sequence
    # offset = (np.random.rand(), np.random.rand())
    # for i in range(num_views):
    #     y, p = sphere_hammersley_sequence(i, num_views, offset)
    #     yaws.append(y)
    #     pitchs.append(p)

    yaws = torch.linspace(0, 2 * 3.1415, total_render_view_num)
    pitchs = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, total_render_view_num))

    # select eval_view_num views
    yaws = yaws[::total_render_view_num // eval_view_num]
    pitchs = pitchs[::total_render_view_num // eval_view_num]

    yaws = yaws.tolist()
    pitchs = pitchs.tolist()
    radius = [2] * eval_view_num
    fov = [40 / 180 * np.pi] * eval_view_num
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        blender_path, '-b', '-P', 
        './dataset_toolkits/blender_script/render.py',
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', engine,  # ('BLENDER_EEVEE_NEXT', 'CYCLES')
    ]
    if no_norm_scene:
        args.append('--no_norm_scene')
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    if BLENDER_EEVEE_NEXT_on_0_device:
        args.insert(1, '--gpu-backend')
        args.insert(2, 'vulkan')
    try:
        if engine == 'CYCLES':
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(cuda_idx)
            if debug:
                call(args, env=env, timeout=timeout_seconds)
            else:
                call(args, env=env, stdout=DEVNULL, timeout=timeout_seconds)
        else:
            if debug:
                call(args, timeout=timeout_seconds)
            else:
                call(args, stdout=DEVNULL, timeout=timeout_seconds)
    except TimeoutExpired:
        print(f"Render timed out for {output_folder} using {engine}")

    return output_folder


# process image
def preprocess_image(input: Image.Image, no_crop=False) -> Image.Image:
    """
    Preprocess the input image.
    """
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        rembg_session = rembg.new_session('u2net')
        output = rembg.remove(input, session=rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    if not no_crop:
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output


def render_image_list(x_0, num_frames=300, resolution=256):
    renderer = OctreeRenderer()
    renderer.rendering_options.resolution = resolution
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 4
    renderer.pipe.primitive = 'voxel'
    
    # Build camera
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()

    exts = []
    ints = []
    for yaw, pitch in zip(yaws, pitch):
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ]).float().cuda() * 2
        fov = torch.deg2rad(torch.tensor(30)).cuda()
        extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        exts.append(extrinsics)
        ints.append(intrinsics)

    images = []
    
    # Build each representation
    x_0 = x_0.cuda()
    assert x_0.shape[0] == 1
    i = 0
    representation = Octree(
        depth=10,
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
        device='cuda',
        primitive='voxel',
        sh_degree=0,
        primitive_config={'solid': True},
    )
    coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
    resolution = x_0.shape[-1]
    representation.position = coords.float() / resolution
    representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

    for _, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
        images.append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))

    return images


class MyDataset(SparseStructureLatentVisMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalization = None
        self.loads = [100 for _ in range(100)]
    
    def __len__(self):
        return 100

    def collate_fn(self, batch):
        return batch


class TrellisEdititngModel:

    def __init__(self, ss_latents_config_path, ss_latents_load_dir, ss_latents_load_ckpt, latents_config_path, latents_load_dir, latents_load_ckpt, load_ema_model_for_inference=False):

        self.ss_latents_config_path = ss_latents_config_path
        self.ss_latents_load_dir = ss_latents_load_dir
        self.ss_latents_load_ckpt = ss_latents_load_ckpt
        self.latents_config_path = latents_config_path
        self.latents_load_dir = latents_load_dir
        self.latents_load_ckpt = latents_load_ckpt

        # load ss_latents model
        config = json.load(open(self.ss_latents_config_path, 'r'))
        cfg = edict()
        cfg.update(config)
        cfg.data_dir = '/home/dataset_model/dataset/3DGS/objaverse_v1' 
        cfg.output_dir = '../../work_dirs/Editing_Training/Debug'
        cfg.load_dir = self.ss_latents_load_dir
        cfg.load_ckpt = self.ss_latents_load_ckpt
        if load_ema_model_for_inference:
            cfg.trainer.args.load_ema_model_for_inference = True

        model_dict = {
            name: getattr(models, model.name)(**model.args).cuda()
            for name, model in cfg.models.items()
        }
        dataset = MyDataset()
        self.ss_latents_trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)
        self.ss_latents_sampler = self.ss_latents_trainer.get_sampler()

        # load latents model
        config = json.load(open(self.latents_config_path, 'r'))
        cfg = edict()
        cfg.update(config)
        cfg.data_dir = '/home/dataset_model/dataset/3DGS/objaverse_v1' 
        cfg.output_dir = '../../work_dirs/Editing_Training/Debug'
        cfg.load_dir = self.latents_load_dir
        cfg.load_ckpt = self.latents_load_ckpt
        if load_ema_model_for_inference:
            cfg.trainer.args.load_ema_model_for_inference = True

        model_dict = {
            name: getattr(models, model.name)(**model.args).cuda()
            for name, model in cfg.models.items()
        }
        self.dataset = MyDataset()
        self.latents_trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)
        self.latents_sampler = self.ss_latents_trainer.get_sampler()

        # ori pipeline
        self.ori_pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.ori_pipeline.cuda()

        # other params
        self.total_steps = 25
        self.gs_bg_color = (0, 0, 0)

    def editing_img_to_ss_latents(self, edited_img):
        pass

    def editing_img_to_latents(self, edited_img):
        pass

    def editing_inference(
        self, 
        ori_ss_latents_path, 
        ori_latents_path, 
        edited_img_path, 
        ori_img_path, 
        vis_resolution=256, 
        ori_latents_norm=False, 
        edited_ss_latents_path=None,
        video_save_path=None, 
        mesh_save_path=None,
        output_video=True,
        output_mesh=True,
        print_time=False,
    ):
        with torch.no_grad():

            assert output_video or output_mesh

            # load img
            processed_ori_img = preprocess_image(Image.open(ori_img_path), no_crop=False)
            processed_ori_img_tensor = torch.from_numpy(np.array(processed_ori_img)).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

            processed_img = preprocess_image(Image.open(edited_img_path), no_crop=False)
            processed_img_tensor = torch.from_numpy(np.array(processed_img)).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

            # load ss_latents
            ori_ss_latent = torch.from_numpy(np.load(ori_ss_latents_path)['mean']).cuda()[None]
            ori_latents_feats = torch.from_numpy(np.load(ori_latents_path)['feats']).cuda()
            if ori_latents_norm:
                std = torch.tensor(self.ori_pipeline.slat_normalization['std'])[None].to(ori_latents_feats.device)
                mean = torch.tensor(self.ori_pipeline.slat_normalization['mean'])[None].to(ori_latents_feats.device)
                ori_latents_feats = (ori_latents_feats - mean) / std
            # print(torch.mean(ori_latents_feats, dim=0), torch.std(ori_latents_feats, dim=0))
            ori_latents_coords = torch.from_numpy(np.load(ori_latents_path)['coords']).cuda()
            if ori_latents_coords.shape[1] == 3:
                ori_latents_coords = torch.cat([torch.zeros_like(ori_latents_coords[:, :1]), ori_latents_coords], dim=1)

            # ss_latents noise
            ss_latents_noise = torch.randn_like(ori_ss_latent)

            # preparing data
            data_tensor = dict()
            data_tensor['ori_cond_img'] = processed_ori_img_tensor
            data_tensor['edited_cond_img'] = processed_img_tensor
            data_tensor['ori_ss_latent'] = ori_ss_latent
            data_tensor['edited_ss_latent'] = ori_ss_latent
            args = self.ss_latents_trainer.get_inference_cond(**data_tensor)
            args['cond'] = torch.clone(args['edited_cond_img'])

            time_a = time.time()
            # inference ss_latents
            if edited_ss_latents_path is None:
                res = self.ss_latents_sampler.sample(
                    self.ss_latents_trainer.models['denoiser'],
                    noise=ss_latents_noise,
                    **args,
                    steps=self.total_steps, 
                    cfg_strength=0,
                    rescale_t=3.0,
                    start_step=0,
                    end_step=self.total_steps,
                    verbose=False,
                ).samples
            else:
                res = torch.from_numpy(np.load(edited_ss_latents_path)['mean']).cuda()[None]
            time_b = time.time()
            if print_time:
                print(f'------SS latents inference time: {time_b - time_a} seconds')

            # get coords
            voxel = self.latents_trainer.dataset.decode_latent(res) > 0
            coords = torch.argwhere(voxel)[:, [0, 2, 3, 4]].int()
            
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], self.latents_trainer.models['denoiser'].in_channels).cuda(),
                coords=coords,
            )
            ori_latents = sp.SparseTensor(
                feats=ori_latents_feats,
                coords=ori_latents_coords.int(),
            )

            # preparing data
            data_tensor['ori_ss_latent'] = ori_latents
            data_tensor['edited_ss_latent'] = ori_latents
            args = self.latents_trainer.get_inference_cond(**data_tensor)
            args['cond'] = torch.clone(args['edited_cond_img'])

            time_c = time.time()
            # inference latents
            slat_no_norm = self.latents_sampler.sample(
                self.latents_trainer.models['denoiser'],
                noise=noise,
                **args,
                steps=self.total_steps, 
                cfg_strength=0,
                rescale_t=3.0,
                start_step=0,
                end_step=self.total_steps,
                verbose=False,
            ).samples
            time_d = time.time()
            if print_time:
                print(f'------Latents inference time: {time_d - time_c} seconds')

            # print(torch.mean(slat_no_norm.feats, dim=0), torch.std(slat_no_norm.feats, dim=0))
            std = torch.tensor(self.ori_pipeline.slat_normalization['std'])[None].to(slat_no_norm.device)
            mean = torch.tensor(self.ori_pipeline.slat_normalization['mean'])[None].to(slat_no_norm.device)
            slat = slat_no_norm * std + mean

            output_format = ['gaussian']
            if output_mesh:
                output_format.append('mesh')

            decoded_slat = self.ori_pipeline.decode_slat(slat, output_format)
            gaussian = decoded_slat['gaussian']

            if output_video:
                # render the edited voxel and gaussian
                voxel_vis = render_image_list(voxel, resolution=vis_resolution)
                gaussian_vis = render_utils.render_video(gaussian[0], resolution=vis_resolution, bg_color=self.gs_bg_color, verbose=False)['color']

                # render the ori voxel and gaussian
                ori_voxel = self.latents_trainer.dataset.decode_latent(ori_ss_latent) > 0
                ori_voxel_vis = render_image_list(ori_voxel, resolution=vis_resolution)
                ori_gaussian = self.ori_pipeline.decode_slat(ori_latents * std + mean, ['gaussian'])['gaussian']
                ori_gaussian_vis = render_utils.render_video(ori_gaussian[0], resolution=vis_resolution, bg_color=self.gs_bg_color, verbose=False)['color']

                processed_ori_img = np.array(processed_ori_img.resize((vis_resolution, vis_resolution), Image.Resampling.LANCZOS))
                processed_img = np.array(processed_img.resize((vis_resolution, vis_resolution), Image.Resampling.LANCZOS))
                vis_images = [np.concatenate([processed_ori_img, ori_voxel_vis[i], ori_gaussian_vis[i], processed_img, voxel_vis[i], gaussian_vis[i]], axis=1) for i in range(len(voxel_vis))]
                imageio.mimsave(video_save_path, vis_images, fps=30)
        
        if output_mesh:
            time_e = time.time()
            # export ori glb
            glb = postprocessing_utils.to_glb(
                gaussian[0],
                decoded_slat['mesh'][0],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
                verbose=False
            )
            glb.export(mesh_save_path)
            time_f = time.time()
            if print_time:
                print(f'------Mesh export time: {time_f - time_e} seconds')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ss_latents_config', type=str, default='ss_flow_img_dit_L_16l8_fp16.json')
    parser.add_argument('--ss_latents_load_id', type=str, default='img_to_voxel')
    parser.add_argument('--ss_latents_load_ckpt', type=int, default=40000)
    parser.add_argument('--latents_config', type=str, default='slat_flow_img_dit_L_64l8p2_fp16.json')
    parser.add_argument('--latents_load_id', type=str, default='voxel_to_texture')
    parser.add_argument('--latents_load_ckpt', type=int, default=40000)
    parser.add_argument('--load_ema_model_for_inference', action='store_true', help='Load ema model for inference')

    parser.add_argument('--save_name', type=str, default='3DEditFormer')
    parser.add_argument('--blender_path', type=str, default='/home/ruihao/install/blender-4.4.3-linux-x64/blender')
    parser.add_argument('--dataset_root_dir', type=str, default='/path_to_3DEditVerse')
    parser.add_argument('--data_info_json_path', type=str, default='dataset_info.json')
    parser.add_argument('--select_json_path', type=str, default='test_data_info.json')
    parser.add_argument('--flux_edit_root_path', type=str, default='flux_edit')
    parser.add_argument('--alpaca_root_path', type=str, default='alpaca')
    parser.add_argument('--mixamo_test_animation', type=list, default=['Michelle', 'Castle Guard 01'])
    parser.add_argument('--mixamo_root_path', type=str, default='mixamo')
    parser.add_argument('--output_video', action='store_true', help='Output video')
    parser.add_argument('--output_mesh', action='store_true', help='Output mesh')
    parser.add_argument('--print_time', action='store_true', help='Print time')

    parser.add_argument('--total_render_view_num', type=int, default=300)
    parser.add_argument('--eval_view_num', type=int, default=10)
    parser.add_argument('--engine', type=str, default='CYCLES', choices=['CYCLES', 'BLENDER_EEVEE_NEXT'])
    parser.add_argument('--BLENDER_EEVEE_NEXT_on_0_device', action='store_true', help='BLENDER_EEVEE_NEXT on 0 device')
    parser.add_argument('--cuda_idx', type=int, nargs='*', default=[0])
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--rank', type=int, default=0, help='Current GPU rank (0 to world_size-1)')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index')
    parser.add_argument('--end_idx', type=int, default=None, help='End index')
    parser.add_argument('--render_timeout_seconds', type=int, default=100, help='Render timeout in seconds')
    parser.add_argument('--debug', action='store_true', help='Debug')
    
    args = parser.parse_args()

    args.data_info_json_path = os.path.join(args.dataset_root_dir, args.data_info_json_path)
    args.select_json_path = os.path.join(args.dataset_root_dir, args.select_json_path)
    args.flux_edit_root_path = os.path.join(args.dataset_root_dir, args.flux_edit_root_path)
    args.alpaca_root_path = os.path.join(args.dataset_root_dir, args.alpaca_root_path)
    args.mixamo_root_path = os.path.join(args.dataset_root_dir, args.mixamo_root_path)

    set_seed(42)

    assert len(args.cuda_idx) == 1

    # load testing data
    save_path = f'./work_dirs/eval_results/{args.save_name}'

    with open(args.select_json_path, 'r') as f:
        select_data = json.load(f)
    with open(args.data_info_json_path, 'r') as f:
        data_info = json.load(f)

    processed_keys = []
    for key, values in select_data.items():
        for value in values:
            processed_keys.append((key, value))

    print('processed keys: ', len(processed_keys))

    if args.start_idx is not None and args.end_idx is not None:
        start_idx = args.start_idx
        end_idx = args.end_idx
        processed_keys = processed_keys[start_idx: end_idx]
    else:
        chunk_size = math.ceil(len(processed_keys) / args.world_size)
        start_idx = args.rank * chunk_size
        end_idx = min((args.rank + 1) * chunk_size, len(processed_keys))
        processed_keys = processed_keys[start_idx: end_idx]
    print(f'World size: {args.world_size}, Rank {args.rank}, tasks index: {start_idx} - {end_idx}')
    
    if args.debug:
        processed_keys = processed_keys[:6]

    trellis_edititng_model = TrellisEdititngModel(
        ss_latents_config_path=f'./configs/editing/{args.ss_latents_config}', 
        ss_latents_load_dir=f'./work_dirs/Editing_Training/{args.ss_latents_load_id}', 
        ss_latents_load_ckpt=args.ss_latents_load_ckpt,
        latents_config_path=f'./configs/editing/{args.latents_config}', 
        latents_load_dir=f'./work_dirs/Editing_Training/{args.latents_load_id}', 
        latents_load_ckpt=args.latents_load_ckpt,
        load_ema_model_for_inference=args.load_ema_model_for_inference,
    )

    print('init trellis edititng model done')   

    render_queue = queue.Queue()
    
    def render_worker():
        while True:
            try:
                task = render_queue.get(timeout=1)
                if task is None:
                    break
                
                mesh_save_path, render_args = task
                
                output_folder = _render(
                    file_path=mesh_save_path,
                    engine=render_args['engine'],
                    cuda_idx=render_args['cuda_idx'],
                    total_render_view_num=render_args['total_render_view_num'],
                    eval_view_num=render_args['eval_view_num'],
                    BLENDER_EEVEE_NEXT_on_0_device=render_args['BLENDER_EEVEE_NEXT_on_0_device'],
                    timeout_seconds=render_args['timeout_seconds'],
                    no_norm_scene=True,
                    debug=render_args['debug'],
                    blender_path=render_args['blender_path'],
                )
                print(f'render done: {mesh_save_path}')
                render_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f'render error: {e}')
                render_queue.task_done()
    
    render_thread = threading.Thread(target=render_worker, daemon=True)
    render_thread.start()
    
    for data in tqdm(processed_keys):

        dataset_type = data[0]
        if dataset_type == 'alpaca':
            key = data[1]
            ori_ss_latents_path = os.path.join(args.alpaca_root_path, data_info[dataset_type][key]['ori_ss_latents_path'])
            ori_latents_path = os.path.join(args.alpaca_root_path, data_info[dataset_type][key]['ori_latents_path'])
            ori_img_path = os.path.join(args.alpaca_root_path, data_info[dataset_type][key]['ori_img_path'])
            edit_img_path = os.path.join(args.alpaca_root_path, data_info[dataset_type][key]['edit_img_path'])
        elif dataset_type == 'flux_edit':
            key = data[1]
            ori_ss_latents_path = os.path.join(args.flux_edit_root_path, data_info[dataset_type][key]['ori_ss_latents_path'])
            ori_latents_path = os.path.join(args.flux_edit_root_path, data_info[dataset_type][key]['ori_latents_path'])
            ori_img_path = os.path.join(args.flux_edit_root_path, data_info[dataset_type][key]['ori_img_path'])
            edit_img_path = os.path.join(args.flux_edit_root_path, data_info[dataset_type][key]['edit_img_path'])
        elif dataset_type == 'mixamo':
            character_name, ori_idx, edit_idx = data[1]
            key = f'{character_name}_{ori_idx}_{edit_idx}'
            ori_ss_latents_path = os.path.join(args.mixamo_root_path, data_info[dataset_type][character_name][ori_idx]['ss_latents_path'])
            ori_latents_path = os.path.join(args.mixamo_root_path, data_info[dataset_type][character_name][ori_idx]['latents_path'])
            ori_img_path = os.path.join(args.mixamo_root_path, data_info[dataset_type][character_name][ori_idx]['img_path'])
            edit_img_path = os.path.join(args.mixamo_root_path, data_info[dataset_type][character_name][edit_idx]['img_path'])
        else:
            raise ValueError(f'Invalid dataset type: {dataset_type}')

        video_save_path = os.path.join(save_path, dataset_type, key, 'video.mp4')
        mesh_save_path = os.path.join(save_path, dataset_type, key, 'edit.glb')
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        if not os.path.exists(mesh_save_path):
            time_a = time.time()
            trellis_edititng_model.editing_inference(
                ori_ss_latents_path=ori_ss_latents_path,
                ori_latents_path=ori_latents_path, 
                edited_img_path=edit_img_path, 
                ori_img_path=ori_img_path,
                output_video=args.output_video,
                output_mesh=args.output_mesh,
                video_save_path=video_save_path,
                mesh_save_path=mesh_save_path,
                ori_latents_norm=True if dataset_type == 'mixamo' else False,
                print_time=args.print_time,
            )
            time_b = time.time()
            if args.print_time:
                print(f'--------- Total Editing inference time: {time_b - time_a} seconds')

        if not os.path.exists(os.path.join(os.path.dirname(mesh_save_path), 'render', 'transforms.json')):
            render_args = {
                'engine': args.engine,
                'cuda_idx': args.cuda_idx[0],
                'total_render_view_num': args.total_render_view_num,
                'eval_view_num': args.eval_view_num,
                'BLENDER_EEVEE_NEXT_on_0_device': args.BLENDER_EEVEE_NEXT_on_0_device,
                'timeout_seconds': args.render_timeout_seconds,
                'debug': args.debug,
                'blender_path': args.blender_path,
            }
            render_queue.put((mesh_save_path, render_args))
            print(f'render task submitted: {mesh_save_path}')

    print('waiting for all render tasks to finish...')
    render_queue.join()

    render_queue.put(None)
    render_thread.join()
    print('all render tasks done')

