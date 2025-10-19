import os
import sys
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
# import torch.multiprocessing as mp # No longer needed for mp.spawn
import numpy as np
import random
# from peft import LoraModel, LoraConfig

from trellis import models, datasets, trainers # Assuming these are your custom modules
from trellis.utils.dist_utils import setup_dist # Assuming this is your custom module


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    # Seed RNGs for reproducibility
    # It's good practice to ensure different ranks get different seeds if necessary,
    # but often a global seed offset by rank is used.
    # The original code used rank, which is fine.
    seed = cfg.get('seed', 42) # Get a base seed from config or use a default
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    # Ensure determinism if desired (can impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(cfg): # local_rank is no longer passed as an argument
    # Set up distributed training using environment variables set by torchrun
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank) # Crucial for torchrun
        # setup_dist will use env:// method or an explicitly passed master_addr/port
        # If setup_dist is designed to use 'env://', master_addr and master_port from cfg might not be needed for it.
        # For safety, we can keep them in cfg if setup_dist needs them,
        # but torchrun sets MASTER_ADDR and MASTER_PORT in the environment.
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12345') # Default if not set
        print(f"Rank {rank}: Initializing distributed training. local_rank={local_rank}, world_size={world_size}, master_addr={master_addr}, master_port={master_port}")
        setup_dist(rank, local_rank, world_size, master_addr, master_port)
    else:
        print("Rank 0: Running in single GPU mode.")
        if torch.cuda.is_available():
             torch.cuda.set_device(local_rank) # for consistency, local_rank is 0

    # Seed rngs
    setup_rng(rank) # Seed with global rank

    if cfg.random_cond_gt:
        cfg.dataset.args.random_cond_gt = True
    if cfg.coords_aug_size is not None:
        cfg.dataset.args.coords_aug_size = cfg.coords_aug_size
    if cfg.feats_aug_grid_size is not None:
        cfg.dataset.args.feats_aug_grid_size = cfg.feats_aug_grid_size
    if cfg.feats_aug_ratio is not None:
        cfg.dataset.args.feats_aug_ratio = cfg.feats_aug_ratio
    if cfg.voxel_aug_ratio is not None:
        cfg.dataset.args.voxel_aug_ratio = cfg.voxel_aug_ratio
    if cfg.adapt_simple_edit_data:
        cfg.dataset.args.adapt_simple_edit_data = True
    if cfg.mixamo_data_repeat_ratio is not None:
        cfg.dataset.args.mixamo_data_repeat_ratio = cfg.mixamo_data_repeat_ratio
    if cfg.random_ori_edit is not None:
        cfg.dataset.args.random_ori_edit = cfg.random_ori_edit
    if cfg.simple_edit_data_if_filtered:
        cfg.dataset.args.simple_edit_data_if_filtered = True
        
    # Load data
    default_data_dir = 'data'  # not used in the dataset init
    dataset = getattr(datasets, cfg.dataset.name)(default_data_dir, **cfg.dataset.args)

    # Build model
    if cfg.ori_ss_latents_weights is not None:
        cfg.models.denoiser.args.ori_ss_latents_weights = cfg.ori_ss_latents_weights
    if cfg.feats_3d_t is not None:
        cfg.models.denoiser.args.feats_3d_t = cfg.feats_3d_t
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda() # .cuda() will use the device set by set_device
        for name, model in cfg.models.items()
    }
    # if cfg.lora:
    #     lora_config = LoraConfig(
    #         r=cfg.lora_rank,
    #         lora_alpha=cfg.lora_alpha,
    #         lora_dropout=cfg.lora_dropout,
    #         target_modules=["to_qkv", "to_out", "to_q", "to_kv"], # "mlp"
    #         exclude_modules='.*editing.*'
    #     )
    #     original_denoiser = model_dict['denoiser']
    #     model_dict['denoiser'] = LoraModel(original_denoiser, lora_config, "default")

    #     for name, param in model_dict['denoiser'].named_parameters():
    #         if 'editing' in name:
    #             param.requires_grad = True

    #     cfg.trainer.args.lora = True

    if cfg.lr is not None:
        cfg.trainer.args.optimizer.args.lr = cfg.lr

    if cfg.batch_size_per_gpu is not None:
        cfg.trainer.args.batch_size_per_gpu = cfg.batch_size_per_gpu

    if cfg.batch_split is not None:
        cfg.trainer.args.batch_split = cfg.batch_split

    if cfg.max_steps is not None:
        cfg.trainer.args.max_steps = cfg.max_steps

    if cfg.train_only_editing_weights:
        for name, param in model_dict['denoiser'].named_parameters():
            if 'editing' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    if cfg.debug:
        cfg.trainer.args.max_steps = 100
        cfg.trainer.args.i_print = 1
        cfg.trainer.args.i_log = 1
        cfg.trainer.args.i_sample = 10
        cfg.trainer.args.i_save = 100
        cfg.trainer.args.init_sample = False
        cfg.trainer.args.init_dataset_vis = False

    if cfg.no_sample_images:
        cfg.trainer.args.no_sample_images = True

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict, dataset, **cfg.trainer.args,
        output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt
    )

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()

    if world_size > 1:
        torch.distributed.barrier() # Ensure all processes finish before exiting
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='/path_to_3DEditVerse/', help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=0, help='Number of retries on error (simplified for torchrun)') # Max retries for main function
    parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
    ## dubug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## training
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size_per_gpu', type=int, default=None, help='Batch size per gpu')
    parser.add_argument('--batch_split', type=int, default=None, help='Batch split')
    parser.add_argument('--max_steps', type=int, default=None, help='Max steps')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--train_only_editing_weights', action='store_true', help='Train only editing weights')
    parser.add_argument('--ori_ss_latents_weights', type=float, default=None, help='Weight for ori ss latents fusing with noising latents')
    parser.add_argument('--feats_3d_t', type=float, nargs=2, default=None, help='Feats 3d t')  # [0.1, 0.9]
    parser.add_argument('--no_sample_images', action='store_true', help='No sample images')
    ## dataset
    parser.add_argument('--random_cond_gt', action='store_true', help='Use random cond gt')
    parser.add_argument('--coords_aug_size', type=int, default=None, help='Coords aug size')
    parser.add_argument('--feats_aug_grid_size', type=int, nargs='*', default=None, help='Feats aug grid size')
    parser.add_argument('--feats_aug_ratio', type=float, nargs=2, default=None, help='Feats aug ratio')
    parser.add_argument('--voxel_aug_ratio', type=float, nargs=2, default=None, help='Voxel aug ratio')
    parser.add_argument('--adapt_simple_edit_data', action='store_true', help='Adapt simple edit data')
    parser.add_argument('--mixamo_data_repeat_ratio', type=float, default=None, help='Mixamo data repeat ratio')
    parser.add_argument('--random_ori_edit', type=float, default=None, help='Random ori edit data')
    parser.add_argument('--simple_edit_data_if_filtered', action='store_true', help='Simple edit data if filtered')
    opt = parser.parse_args()

    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    # opt.num_gpus is not used to launch processes anymore with torchrun.
    # It can be kept if cfg.num_gpus is used elsewhere in the logic,
    # otherwise, it's informational.
    # The actual number of GPUs used per node is determined by `torchrun --nproc_per_node`.

    ## Load config
    config_from_file = json.load(open(opt.config, 'r'))
    for dataset_replace_key in [
        'mixamo_root_path', 'flux_edit_root_path', 'alpaca_root_path', 'dataset_info_path', 
        'alpaca_confidence_json_path', 'flux_edit_confidence_json_path', 'flux_edit_alpaca_test_json_path'
    ]:
        config_from_file['dataset']['args'][dataset_replace_key] = config_from_file['dataset']['args'][dataset_replace_key].replace('/path_to_3DEditVerse', opt.data_dir)
    
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__) # Command line args take precedence
    cfg.update(config_from_file) # Then update with file config (potentially overwriting CLI defaults if not specified in CLI)
    # To ensure CLI overrides file config for shared keys:
    # temp_cfg = edict(config_from_file)
    # temp_cfg.update(opt.__dict__) # CLI overrides file
    # cfg = temp_cfg

    # Update cfg with command line arguments again to ensure they have priority
    # This makes CLI args override json config values.
    for key, value in opt.__dict__.items():
        # Only update if the arg was actually provided or is not the default for action='store_true'
        if value is not None:
             # For argparse arguments with defaults, they will always be in opt.__dict__.
             # For 'action=store_true', default is False. If specified, it's True.
             # This logic ensures CLI args effectively override config file values.
            is_default_argparse = False
            for action in parser._actions:
                if action.dest == key:
                    if action.default == value and not isinstance(action, argparse._StoreTrueAction) and not isinstance(action, argparse._StoreFalseAction):
                        # Check if the value is the default AND it wasn't explicitly set by user
                        # This part is tricky without checking sys.argv. A simpler approach is just to override.
                        pass # Simpler to just let CLI override.
            cfg[key] = value


    # Get rank for file operations (like saving config)
    # These env vars are set by torchrun.
    # Use 0 if not in a distributed environment (e.g. world_size=1)
    current_rank = int(os.environ.get('RANK', 0))
    world_size_for_setup = int(os.environ.get('WORLD_SIZE', 1))

    if current_rank == 0: # Only master process should create dirs and save initial files
        print('\n\nConfig:')
        print('=' * 80)
        # Use a serializable dictionary for printing/saving
        # edict can sometimes have issues with json.dump if it contains non-standard types.
        config_to_print_save = dict(cfg)
        print(json.dumps(config_to_print_save, indent=4))

        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp) # This will show the torchrun command if applicable
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config_to_print_save, fp, indent=4)

    # Run
    # The auto_retry logic needs to be within the main call if it's for application-level retries.
    # If it was for process launch failures, torchrun/scheduler handles that.
    # For simplicity, if an error occurs in `main` and `auto_retry` is > 0, we can try rerunning `main`.
    # Note: This simplistic retry doesn't reset CUDA state or other global states perfectly.
    # A more robust retry would be at the job submission level.

    cfg = find_ckpt(cfg) # Find checkpoint before potentially entering retry loop

    if cfg.auto_retry > 0 and world_size_for_setup > 1: # Only attempt retry if distributed
        print(f"Warning: auto_retry ({cfg.auto_retry}) within torchrun script has limited effect and might not recover from all errors. Job-level retry is preferred.")

    for rty in range(cfg.auto_retry + 1): # +1 because range is exclusive at the end, so 0 retries means 1 attempt
        try:
            # cfg = find_ckpt(cfg) # Moved outside loop; typically don't want to re-find checkpoint on retry unless intended
            main(cfg)
            break # Success
        except Exception as e:
            print(f"Error during main execution: {e}")
            if rty < cfg.auto_retry:
                print(f"Retrying ({rty + 1}/{cfg.auto_retry})...")
                if world_size_for_setup > 1: # If distributed, wait a bit before retrying
                    torch.distributed.barrier() # Wait for all processes to hit the error before retrying
                    # A small delay might be useful in some cases
                    # import time
                    # time.sleep(5)
            else:
                print("Max retries reached. Failing.")
                if world_size_for_setup > 1 and torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                raise e # Re-raise the exception if max retries are exhausted
    main(cfg)
    
