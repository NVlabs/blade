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



import argparse, time
import os, numpy as np
import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from blade.datasets.builder import build_dataloader, build_dataset
from blade.models.architectures.builder import build_architecture


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir',
                        default=None,
                        help='the dir to save evaluation results')
    parser.add_argument('--checkpoint',
                        help='checkpoint file',
                        default='',
                        type=str)
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--num-data', type=int, default=0)
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['pa-mpjpe', 'mpjpe', 'pve', 'z_error',
                 'raw_miou', 'raw_pmiou', 'opti_miou', 'opti_pmiou', 'opti_miou_w_gt_mask', 'opti_pmiou_w_gt_mask',
                 'inv_z_error', 'xy_error', 'f_perc_error'],  # 'miou' pmiou
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M')
    parser.add_argument('--gpu_collect',
                        action='store_true',
                        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    parser.add_argument('--data-name',
                        type=str,
                        default='pdhuman',
                        help='dataset used for testing')
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    if isinstance(cfg.data.test, dict):
        cfg.data.test = cfg.data.test[args.data_name]

    if args.num_data:
        cfg.data.test.update(num_data=args.num_data)

    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=cfg.data.samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False,
                                   round_up=False)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        print('No checkpoint provided, will use pretrained model in config')
        model.init_weights()
    # model.load_separate_pretrained_depthnet()
    print(f"args.metrics: {args.metrics}")
    use_miou = 'raw_miou' in args.metrics
    use_pmiou = 'raw_pmiou' in args.metrics
    model.miou = use_miou
    model.pmiou = use_pmiou
    model.vis = args.vis
    print(f'model.miou is {model.miou}, model.pmiou is {model.pmiou}')
    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        print('Start single gpu test')
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    rank, _ = get_dist_info()
    print(f'\nrank is: {rank}')
    eval_cfg = cfg.get('evaluation', args.eval_options)
    eval_cfg.update(dict(metric=args.metrics))

    # time.sleep(60)
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'\n------------------------------')
    print(f'DATASET NAME: {args.data_name}')
    print(f'CHECKPOINTS: {args.checkpoint}')
    if rank == 0:
        if args.work_dir:
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        results, all_results = dataset.evaluate(outputs, args.work_dir, **eval_cfg)
        version = args.checkpoint.split("/")[-2]
        epoch = args.checkpoint.split("/")[-1].split(".")[0]
        root_dir = (osp.dirname(osp.dirname(osp.abspath(__file__)))
                    + f'/{args.data_name}_{version}_{epoch}_results.npy')
        np.save(root_dir, all_results)
        # a = np.load(root_dir, allow_pickle=True)

        for k, v in results.items():
            print(f'\n------- {k} : {v:.3f} ----------')

        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
