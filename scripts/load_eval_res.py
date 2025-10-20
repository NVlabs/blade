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



import argparse, time, json
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
    parser.add_argument('--res_in', help='input intermmedaite result file')
    parser.add_argument('--out', help='out result file')
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
    rank, _ = get_dist_info()
    eval_cfg = cfg.get('evaluation', args.eval_options)
    eval_cfg.update(dict(metric=args.metrics))

    if args.work_dir:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    res = json.load(open(args.res_in))
    results, all_results = dataset.evaluate(None, args.work_dir, res=res, **eval_cfg)

    for k, v in results.items():
        print(f'\n------- {k} : {v:.3f} ----------')

    if args.out and rank == 0:
        out_fn = os.path.join(args.work_dir, "results.pth")
        print(f'\nwriting results to {out_fn}')
        mmcv.dump(results, out_fn)


if __name__ == '__main__':
    main()
