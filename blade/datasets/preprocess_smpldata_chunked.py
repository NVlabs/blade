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


print("hi")
import argparse, socket, time, sys, math, os
conda_env = os.getenv('CONDA_DEFAULT_ENV')
print(f"Active Conda environment: {conda_env}")
from PIL import Image
import torch, numpy as np, os, time
from os.path import abspath, dirname
# from blade.configs.base import root
from glob import glob

from gitdb.fun import chunk_size

# from blade.models.body_models.smpl_x import SMPLX
from blade.models.body_models.builder import build_body_model
from blade.configs.base import body_model_train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../smplx_repo')))
from transfer_model.optimizers import build_optimizer, minimize
from transfer_model.transfer_model import run_fitting
import smplx
from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle
from utils import init_conversion
from gender_lookup import *

import matplotlib.pyplot as plt

print("finished import")

repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(repo_root, 'mmhuman_data')

H36M = 'h36m'
PDHUMAN = 'pdhuman'
HUMMAN = 'humman'
all_datasets = {H36M: 'h36m_mosh_train_transl.npz', PDHUMAN: 'pdhuman_train.npz', HUMMAN: 'humman_train.npz'}
device = torch.device('cpu')
# device = torch.device('cuda')

DO_VIS = False

cur_dataset = DATASET = sys.argv[1]
PROCESS_ID = int(sys.argv[2])
SEQ_START = int(sys.argv[3])
SEQ_END = int(sys.argv[4])

print("before checks")

if len(sys.argv) != 5:
    assert False, f"usage: python preprocess_smpldata_chunked.py <DATASET> <PROCESS_ID> <SEQ_START> <SEQ_END>"
if DATASET not in [H36M, PDHUMAN, HUMMAN]:
    assert False, f"{DATASET} is not a valid dataset, choose from [{H36M}, {PDHUMAN}, {HUMMAN}]"




output_chunk_folder = f'{data_root}/preprocessed_datasets/{cur_dataset}_smplx_chunks/'
os.makedirs(output_chunk_folder, exist_ok=True)
output_fn = f'{output_chunk_folder}/{all_datasets[cur_dataset].replace(".npz", f"_smplx_{PROCESS_ID}_{SEQ_START}_{SEQ_END}.npz")}'
model_path = f'{dirname(dirname(dirname(abspath(__file__))))}/body_models/'
smplx_model_neutral = smplx.create(model_path, model_type='smplx', gender='neutral', use_pca=False).to(device)
verify_smpl_model = build_body_model(body_model_train).to(device)

smpl_repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../smplx_repo')
cfg, destination_model, def_matrix, mask_ids = init_conversion(smpl_repo_path, device, 'smpl', 'smplx')

seq_cnt_i = 0
t0 = time.time()

ann_file_fn = f'{data_root}/preprocessed_datasets/{all_datasets[cur_dataset]}'
ann_file = np.load(ann_file_fn, allow_pickle=True)
image_root = f'{data_root}/datasets/{cur_dataset}'

total_samples = len(ann_file["image_path"])
print(f'a total of {total_samples} images')

smpl_params = ann_file['smpl'].item()
smpl_body_pose = smpl_params['body_pose']
smpl_glob_ori = smpl_params['global_orient']
smpl_betas = smpl_params['betas']
smpl_transl = smpl_params['transl']
# cam_params = ann_file['cam_param'].item().generate_cameras_dict()
# focal_list = ann_file['focal_length']
K_list = ann_file['K']

cnt_i = 0
out = {'transl': [], 'global_orient': [], 'body_pose': [], 'betas': [],
       'left_hand_pose': [], 'right_hand_pose': [], 'img_i': []}

# read existing
print("check existing results")
if os.path.exists(output_fn):
    print(f"there're previous results: {output_fn}")
    existing_data = torch.load(output_fn)
    out['betas'] = existing_data['betas']
    out['body_pose'] = existing_data['body_pose']
    out['transl'] = existing_data['transl']
    out['global_orient'] = existing_data['global_orient']
    out['left_hand_pose'] = existing_data['left_hand_pose']
    out['right_hand_pose'] = existing_data['right_hand_pose']
    out['img_i'] = existing_data['img_i']

    assert (len(out['betas']) == len(out['body_pose']) == len(out['transl']) == len(out['global_orient'])
            == len(out['left_hand_pose']) == len(out['right_hand_pose']) == len(out['img_i'])), (f"PID: {PROCESS_ID}, "
             f"loaded existing files, but inconsistent number of items in the lists: "
             f"{len(out['betas'])} betas, {len(out['body_pose'])} body_pose, {len(out['transl'])} transls, "
             f"{len(out['global_orient'])} global_orients, {len(out['left_hand_pose'])} left_hand_pose, "
             f"{len(out['right_hand_pose'])} right_hand_pose, {len(out['img_i'])} img_i")

    SEQ_START = out['img_i'][-1] + 1
    cnt_i += len(out['betas'])
    print(f"loaded {cnt_i} samples")

print(f"start from {SEQ_START}, end at {min(total_samples, SEQ_END+1)}")
for seq_i in range(SEQ_START, min(total_samples, SEQ_END+1)):
    cnt_i += 1

    cur_smpl_body_pose = torch.tensor(smpl_body_pose[seq_i])[None].float().to(device)
    cur_smpl_glob_ori = torch.tensor(smpl_glob_ori[seq_i])[None, None].float().to(device)
    cur_smpl_betas = torch.tensor(smpl_betas[seq_i])[None].float().to(device)
    cur_smpl_transl = torch.tensor(smpl_transl[seq_i])[None].float().to(device)

    print(f'\n\nchunk {PROCESS_ID}, seq {seq_cnt_i} frame {cnt_i}, {time.time()-t0} seconds so far')

    #  load images
    img_fn = f'{data_root}/datasets/{cur_dataset}/{ann_file["image_path"][seq_i]}'
    if cur_dataset == H36M:
        img_fn = img_fn.replace('/images/', '/')
    im = Image.open(img_fn).convert('RGB')
    # plt.imshow(im); plt.show()
    h, w = im.height, im.width


    # load data
    # seq = ann_file["image_path"][seq_i].split('/')[0]
    # cam_id = ann_file["image_path"][seq_i].split('/')[2].split('_')[-1].split('.')[1]
    # cam_params[(seq, cam_id)]['rotation_mat']

    verify_out = verify_smpl_model(betas=cur_smpl_betas,
                                   body_pose=cur_smpl_body_pose,
                                   global_orient=cur_smpl_glob_ori,
                                   transl=cur_smpl_transl,
                                   return_verts=True)
    verify_vert = verify_out['vertices']
    verify_pelvis = verify_out['joints'][0, 0, :, None]

    smpl_mesh = {'vertices': verify_vert,
                  'faces': verify_smpl_model.faces[None]}

    convert_out = run_fitting(cfg, smpl_mesh, destination_model, def_matrix, mask_ids)

    out['betas'].append(convert_out['betas'].float())
    out['body_pose'].append(matrix_to_axis_angle(convert_out['body_pose']).float())
    out['transl'].append(convert_out['transl'].float())
    out['global_orient'].append(matrix_to_axis_angle(convert_out['global_orient']).float())
    out['left_hand_pose'].append(matrix_to_axis_angle(convert_out['left_hand_pose']).float())
    out['right_hand_pose'].append(matrix_to_axis_angle(convert_out['right_hand_pose']).float())
    out['img_i'].append(seq_i)

    torch.save(out, output_fn)
    #  render SMPLX & SMPL
    if DO_VIS:
        # cam -> image projection
        K = torch.tensor(K_list[seq_i]).float().to(device)
        verify_2D = (K @ verify_vert[0].t()).detach().cpu()
        verify_2D[0] /= verify_2D[2]
        verify_2D[1] /= verify_2D[2]
        plt.imshow(im); plt.scatter(verify_2D[0], verify_2D[1], c='g', s=0.02)

        output = smplx_model_neutral(betas=out['betas'][-1], body_pose=out['body_pose'][-1],
                                     left_hand_pose=out['left_hand_pose'][-1], right_hand_pose=out['right_hand_pose'][-1],
                                     global_orient=out['global_orient'][-1], transl=out['transl'][-1], return_verts=True)
        vertices = output.vertices
        pelvis = output.joints[0,0,:,None]

        verify_2D = (K @ vertices[0].t()).detach().cpu()
        verify_2D[0] /= verify_2D[2]
        verify_2D[1] /= verify_2D[2]
        plt.imshow(im); plt.scatter(verify_2D[0], verify_2D[1], c='r', s=0.01); plt.show()
        time.sleep(0.1)

t1 = time.time()

print(f'total time: {t1-t0}, saved to {output_fn}')
