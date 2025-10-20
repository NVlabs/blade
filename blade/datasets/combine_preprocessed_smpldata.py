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


import argparse, socket, time, sys, math, re
from PIL import Image
import torch, numpy as np, os, time
from os.path import abspath, dirname
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../smplx_repo')))

import matplotlib.pyplot as plt

repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(repo_root, 'mmhuman_data')

H36M = 'h36m'
PDHUMAN = 'pdhuman'
HUMMAN = 'humman'
all_datasets = {H36M: 'h36m_mosh_train_transl.npz', PDHUMAN: 'pdhuman_train.npz', HUMMAN: 'humman_train.npz'}
device = torch.device('cpu')
# device = torch.device('cuda')


DO_VIS = False


cur_dataset = sys.argv[1]
if cur_dataset not in [H36M, PDHUMAN, HUMMAN]:
    assert False, f"{cur_dataset} is not a valid dataset, choose from [{H36M}, {PDHUMAN}, {HUMMAN}]"

def extract_second_number(filename):
    # Use regex to extract all the numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # Return the second number (third-last number in terms of the filename's format)
    return int(numbers[-2])

converted_fn_list = sorted(glob(f'{data_root}/preprocessed_datasets/{cur_dataset}/{all_datasets[cur_dataset].replace(".npz", "")}_smplx_chunks/'
                                f'{all_datasets[cur_dataset].replace(".npz", "")}_smplx_*_*_*.npz'), key=extract_second_number)

ann_file_fn = f'{data_root}/preprocessed_datasets/{all_datasets[cur_dataset]}'
ann_file = dict(np.load(ann_file_fn, allow_pickle=True))
out_fn = ann_file_fn.replace('.npz', '_smplx.npz')

n_samples = len(ann_file["image_path"])
print(f'a total of {n_samples} images')


cnt_i = 0
ann_file['smplx'] = {'transl': [], 'global_orient': [], 'body_pose': [], 'betas': [],
       'left_hand_pose': [], 'right_hand_pose': [], 'img_i': []}
for converted_fn in converted_fn_list:
    cnt_i += 1
    converted_data = torch.load(converted_fn)

    # assert converted_data["img_i"] == cnt_i, f"data id ({converted_data["img_i"]}) different from count {cnt_i}"
    ann_file['smplx']['transl'].extend(converted_data["transl"])
    ann_file['smplx']['global_orient'].extend(converted_data["global_orient"])
    ann_file['smplx']['body_pose'].extend(converted_data["body_pose"])
    ann_file['smplx']['betas'].extend(converted_data["betas"])
    ann_file['smplx']['left_hand_pose'].extend(converted_data["left_hand_pose"])
    ann_file['smplx']['right_hand_pose'].extend(converted_data["right_hand_pose"])
    ann_file['smplx']['img_i'].extend(converted_data["img_i"])

    start_i = int(converted_fn.split('_')[-2])
    end_i = int(converted_fn.split('_')[-1].replace('.npz', ''))-1
    f_i = 0
    for cur_i in converted_data["img_i"]:
        if cur_i != (f_i+start_i):
            print(f"{converted_fn.split('/')[-1]} out of order: should be {f_i+start_i} but is {cur_i}")
        f_i += 1

    n_added = len(converted_data["transl"])
    assert (len(converted_data['betas']) == len(converted_data['body_pose']) == len(converted_data['transl'])
            == len(converted_data['global_orient']) == len(converted_data['left_hand_pose'])
            == len(converted_data['right_hand_pose']) == len(converted_data['img_i'])), (f"inconsistent number of items in the lists: "
             f"{len(converted_data['betas'])} betas, {len(converted_data['body_pose'])} body_pose, {len(converted_data['transl'])} transls, "
             f"{len(converted_data['global_orient'])} global_orients, {len(converted_data['left_hand_pose'])} left_hand_pose, "
             f"{len(converted_data['right_hand_pose'])} right_hand_pose, {len(converted_data['img_i'])} img_i")

# # !!!!!!!!!!!! for local debug only!!!!!!!!!!!!!!!!!
# for key in ann_file:
#     if key in ['image_path', 'image_size', 'bbox_xywh', 'K', 'focal_length',
#                'keypoints2d', 'keypoints3d', 'distortion_max']:
#         ann_file[key] = ann_file[key][:n_added]
#     elif key == 'cam_params':
#         for subkey in ['R', 'K', 'T']:
#             ann_file[key].item()[subkey] = ann_file[key].item()[subkey][:n_added]
#     elif key == 'smpl':
#         for subkey in ['betas', 'global_orient', 'body_pose', 'transl']:
#             ann_file[key].item()[subkey] = ann_file[key].item()[subkey][:n_added]
# ann_file['__data_len__'] = len(ann_file["image_path"])

n_converted_samples = len(ann_file["image_path"])
print(f'converted to a total of {n_converted_samples} images')

np.savez_compressed(out_fn, **ann_file)
assert len(ann_file['smplx']['transl']) == n_samples, f"ERROR: combined {len(ann_file['smplx']['transl'])} samples, but there should be {n_samples}."
assert ann_file['smplx']['img_i'][-1] == n_samples-1, f"ERROR: last img idx ({ann_file['smplx']['img_i'][-1]}) should be {n_samples-1}."


