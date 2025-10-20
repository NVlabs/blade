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


import numpy as np
import torch.nn.functional as F


def select_frames_index(
    index: int,
    batch_size: int,
    num_frames: int,
    interval: int,
    fix_interval: bool = True,
    temporal_successive: bool = True,
):
    """Select two batches of frame index.
        If temporal_successive: the source and target indexes will be
            successive, else will be random.
        If fix_interval: the source and target indexes will have a fixed
            interval, else will be from -interval to interval.
    """
    if temporal_successive:
        index_0_source = index
        if fix_interval:
            index_0_target = min(index_0_source + interval,
                                 num_frames - batch_size + 1)
        else:
            index_0_target = np.random.randint(
                low=max(0, index_0_source - interval),
                high=min(num_frames - batch_size + 1,
                         index_0_source + interval))
        indexes_source = np.arange(index_0_source, index_0_source + batch_size)
        indexes_target = np.arange(index_0_target, index_0_target + batch_size)
    else:
        indexes_source = np.random.randint(low=0,
                                           high=num_frames,
                                           size=batch_size)
        if fix_interval:
            indexes_target = indexes_source + interval
            indexes_target = np.clip(indexes_target, 0, num_frames - 1)
        else:
            indexes_target = np.random.randint(
                low=-interval, high=interval, size=batch_size) + indexes_source
            indexes_target = np.clip(indexes_target, 0, num_frames - 1)
    return list(indexes_source), list(indexes_target)


def pad_to_square(image):
    # Assume the image is a 3D tensor with shape (C, H, W)
    _, height, width = image.shape

    # Calculate padding amounts
    if height == width:
        return image
    elif height > width:
        padding = (height - width) // 2
        padding_left = padding
        padding_right = height - (padding_left + width)
        padding_top = 0
        padding_bottom = 0
    else:
        padding = (width - height) // 2
        padding_top = padding
        padding_bottom = width - (padding_top + height)
        padding_left = 0
        padding_right = 0

    # Apply padding
    padded_image = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom))

    return padded_image


from smplx import build_layer
import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../smplx_repo/transfer_model')))
from smplx_repo.transfer_model.config.defaults import conf as default_conf
from smplx_repo.transfer_model.utils import read_deformation_transfer
# from transfer_model.optimizers import build_optimizer, minimize
# from transfer_model.transfer_model import run_fitting
from omegaconf import OmegaConf
import torch, os.path as osp
from loguru import logger

def init_conversion(smpl_repo_path, device, from_model, to_model):
    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(f'{smpl_repo_path}/config_files/{from_model}2{to_model}.yaml'))
    cfg.body_model.folder = f'{smpl_repo_path}/../body_models'
    cfg.deformation_transfer_path = f'{smpl_repo_path}/../pretrained/transfer_data/{from_model}2{to_model}_deftrafo_setup.pkl'

    body_model = build_layer(cfg.body_model.folder, cfg.body_model.model_type, **cfg).to(device)
    deformation_transfer_path = cfg.get('deformation_transfer_path', '')

    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    mask_ids_fname = osp.expandvars(cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    return cfg, body_model, def_matrix, mask_ids
