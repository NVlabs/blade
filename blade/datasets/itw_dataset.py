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

import glob, torch, copy
from abc import ABCMeta
import numpy as np
import cv2
from mmhuman3d.data.datasets.base_dataset import BaseDataset
from blade.datasets.pipelines.compose import Compose
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class ITWDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 batch_list,
                 pipeline):
        self.batch_list = batch_list
        self.id_list = list(self.batch_list.keys())

        self.pipeline_load = Compose(pipeline[:1])
        Collect_idx = None
        for i in range(1, len(pipeline)):
            if pipeline[i]['type'] == 'Collect':
                Collect_idx = i+1
                break
        self.pipeline_depth = Compose(pipeline[1:Collect_idx])
        self.pipeline_pose = Compose(pipeline[Collect_idx:])

    def __len__(self):
        return len(self.batch_list)

    def prepare_raw_data(self, idx):
        sample_idx = idx
        # idx = int(self.valid_index[idx])

        info = {}
        cur_id = self.id_list[idx]
        cur_data = self.batch_list[cur_id]
        info['id'] = cur_id
        info['image_path'] = cur_data['rgb_file']
        info['mask_path'] = cur_data.get('alpha_file', None)
        info['img_prefix'] = None
        # orig_img = cv2.imread(self.image_paths[idx])
        orig_img = Image.open(cur_data['rgb_file'])
        info['full_size_img'] = orig_img
        resized_img = np.asarray((self.resize_and_crop(orig_img, max_side_length=1280)))


        if len(resized_img.shape) == 3:
            info['img'] = resized_img[:,:,[2,1,0]]
        else:
            info['img'] = resized_img[:,:,None].repeat(3,-1)

        h, w, _ = info['img'].shape
        info['scale'] = np.array([max(h, w), max(h, w)])

        info['ori_shape'] = (h, w)
        info['center'] = np.array([w / 2, h / 2])

        info['sample_idx'] = sample_idx
        info['has_focal_length'] = 0
        info['has_transl'] = 0
        info['has_K'] = 0
        info['K'] = np.eye(3, 3).astype(np.float32)


        return info

    def evaluate(self, ):
        raise NotImplementedError

    def resize_and_crop(self, image, max_side_length=512):
        """
        Square-pad, resize to max_side_length, then center-crop so both dims are divisible by 16.

        Args:
            image (PIL.Image.Image): The input image.
            max_side_length (int): Target square size before 16-divisible crop.

        Returns:
            PIL.Image.Image: Square image, side length divisible by 16.
        """
        # 1) Square-pad (centered) to max(width, height). Pad color = black.
        w, h = image.size
        if w != h:
            side = max(w, h)
            pad_w = side - w
            pad_h = side - h
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

        # 2) Resize to max_side_length × max_side_length
        if image.size != (max_side_length, max_side_length):
            image = image.resize((max_side_length, max_side_length), Image.LANCZOS)

        # 3) Center-crop so side length is divisible by 16
        final_side = (max_side_length // 16) * 16
        # Guard against very small max_side_length
        final_side = max(16, final_side)

        if final_side < max_side_length:
            offset = (max_side_length - final_side) // 2
            image = image.crop((
                offset,  # left
                offset,  # top
                offset + final_side,  # right
                offset + final_side  # bottom
            ))

        return image

    def __getitem__(self, idx: int):
        return self.prepare_data(idx)

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        data = self.pipeline_load(info)
        data_depth = copy.deepcopy(data)
        orig_img = torch.tensor(data['img']).permute(2, 0, 1)

        data = self.pipeline_pose(data)
        data['posenet_img'] = data['img']

        # seg mask
        if info['mask_path'] is not None:
            seg_mask = torch.tensor(np.asarray(Image.open(info['mask_path'])))[..., -1]
            data['seg_masks'] = seg_mask[None]
        # else:
        #     seg_mask = ((data['posenet_img'][0] > -2.1179) & (data['posenet_img'][1] > -2.0357)
        #                 & (data['posenet_img'][2] > -1.8044)) * 1.
        #     data['seg_masks'] = None

        data['is_demo'] = True
        data['id_idx'] = idx
        return data