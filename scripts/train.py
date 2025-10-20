# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
os.environ['PYTORCH_SHARING_STRATEGY'] = 'file_system'   # must be set before importing torch
# optionally pick a large, writable tmpdir (avoid /dev/shm)
# os.environ.setdefault('TMPDIR', '/tmp')

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

import argparse, copy, glob, os, re, sys, random, shutil
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.cnn import MODELS

from blade.models.architectures.builder import build_architecture
from blade.datasets.builder import build_dataset
from blade.utils.helpers import get_global_rank

from mmhuman3d import __version__
from mmhuman3d.apis import set_random_seed, train_model

from mmhuman3d.utils.collect_env import collect_env
from mmhuman3d.utils.logger import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use '
                            '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='arguments in dict')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_architecture(cfg.model)
    model.init_weights()
    # if args.use_half_precision:
    #     model.half()
    if distributed and cfg.use_syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("converted to SyncBatchNorm")
    else:
        print("not converted to SyncBatchNorm")
    model.vis = False
    model.miou = 'miou' in cfg.evaluation['metric']
    model.pmiou = 'pmiou' in cfg.evaluation['metric']

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmhuman3d_version=__version__)

    def extract_epoch_number(filename):
        match = re.search(r'epoch_(\d+).pth', filename)
        if match:
            return int(match.group(1))
        return -1

    if args.resume_from is not None:
        ckpt_list = glob.glob(os.path.join(cfg.work_dir, 'epoch_*.pth'))
        if len(ckpt_list) != 0:
            ckpt_list.sort(key=extract_epoch_number)
            cfg.resume_from = ckpt_list[-1]

    # save config file
    if get_global_rank() == 0:
        repo_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(args.config))))
        blade_fn = osp.join(repo_root, 'blade', 'models', 'architectures', 'blade.py')
        print(f"copying {blade_fn}")
        shutil.copy(blade_fn, args.work_dir)
        base_fn = osp.join(repo_root, 'blade', 'configs', 'base.py')
        print(f"copying {base_fn}")
        shutil.copy(base_fn, args.work_dir)
        print(f"copying {args.config}")
        shutil.copy(args.config, args.work_dir)

    train_model(model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                device='cpu' if args.device == 'cpu' else 'cuda',
                meta=meta)


if __name__ == '__main__':
    main()
