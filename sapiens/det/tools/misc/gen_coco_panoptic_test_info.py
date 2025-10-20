# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling BLADE or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp

from mmengine.fileio import dump, load


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate COCO test image information '
        'for COCO panoptic segmentation.')
    parser.add_argument('data_root', help='Path to COCO annotation directory.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data_root = args.data_root
    val_info = load(osp.join(data_root, 'panoptic_val2017.json'))
    test_old_info = load(osp.join(data_root, 'image_info_test-dev2017.json'))

    # replace categories from image_info_test-dev2017.json
    # with categories from panoptic_val2017.json which
    # has attribute `isthing`.
    test_info = test_old_info
    test_info.update({'categories': val_info['categories']})
    dump(test_info, osp.join(data_root,
                             'panoptic_image_info_test-dev2017.json'))


if __name__ == '__main__':
    main()
