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
# from zolly.configs.base import root
from glob import glob
from tqdm import tqdm

from gitdb.fun import chunk_size

# from zolly.models.body_models.smpl_x import SMPLX
from blade.models.body_models.builder import build_body_model
from blade.configs.base import body_model_train
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../smplx_repo')))
from smplx_repo.transfer_model.optimizers import build_optimizer, minimize
from smplx_repo.transfer_model.transfer_model import run_fitting
import smplx
from pytorch3d.transforms import euler_angles_to_matrix, axis_angle_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle
from utils import init_conversion
from gender_lookup import *

import matplotlib.pyplot as plt


repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(repo_root, 'mmhuman_data')

label_fn = 'bedlamcc.npz'
device = torch.device('cpu')
# device = torch.device('cuda')


DO_VIS = False

def extract_second_number(filename):
    # Use regex to extract all the numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # Return the second number (third-last number in terms of the filename's format)
    return int(numbers[-2])

converted_fn_list = sorted(glob(os.path.join(data_root, 'preprocessed_datasets/bedlamcc_smplx_chunks',
                                f'{label_fn.replace(".npz", "")}_*_*_*.npz')), key=extract_second_number)

save_fn = f'{data_root}/preprocessed_datasets/{label_fn}'


cnt_i = 0
out = {'img_fn': [], 'sample_idx': [],
        'smplx':{
            'body_pose': [], 'left_hand_pose': [], 'right_hand_pose': [],
            'global_orient': [], 'betas': [], 'transl': [],
           },
       'focal_length': [], 'K': [], 'ori_shape': [], 'center': [], 'scale': [], 'gender': []}

with tqdm(total=len(converted_fn_list), desc="Progress", leave=True) as pbar:
    for converted_fn in converted_fn_list:
        cnt_i += 1
        converted_data = torch.load(converted_fn)

        # assert converted_data["img_i"] == cnt_i, f"data id ({converted_data["img_i"]}) different from count {cnt_i}"
        # out['sequence_name'].extend(converted_data["sequence_name"])
        out['img_fn'].extend(converted_data["img_fn"])
        # out['image_name'].extend(converted_data["image_name"])
        # out['has_keypoints2d'].extend(converted_data["has_keypoints2d"])
        out['focal_length'].extend(converted_data["focal_length"])
        # out['ori_shape'].extend(converted_data["ori_shape"])
        out['center'].extend(converted_data["center"])
        out['scale'].extend(converted_data["scale"])

        out['gender'].extend(converted_data["gender"])
        out['smplx']['body_pose'].extend(converted_data['smplx']["body_pose"])
        out['smplx']['left_hand_pose'].extend(converted_data['smplx']["left_hand_pose"])
        out['smplx']['right_hand_pose'].extend(converted_data['smplx']["right_hand_pose"])
        out['smplx']['global_orient'].extend(converted_data['smplx']["global_orient"])
        out['smplx']['betas'].extend(converted_data['smplx']["betas"])
        out['smplx']['transl'].extend(converted_data['smplx']["transl"])

        # start_i = int(converted_fn.split('_')[-2])
        # end_i = int(converted_fn.split('_')[-1].replace('.npz', ''))-1
        # f_i = 0
        # for cur_i in converted_data["img_i"]:
        #     if cur_i != (f_i+start_i):
        #         print(f"{converted_fn.split('/')[-1]} out of order: should be {f_i+start_i} but is {cur_i}")
        #     f_i += 1

        n_added = len(converted_data['smplx']["transl"])
        assert (len(converted_data['smplx']['betas']) == len(converted_data['smplx']['body_pose']) == len(converted_data['smplx']['transl'])
                == len(converted_data['smplx']['global_orient']) == len(converted_data['smplx']['left_hand_pose'])), (f"inconsistent number of items in the lists: "
                 f"{len(converted_data['smplx']['betas'])} betas, {len(converted_data['smplx']['body_pose'])} body_pose, {len(converted_data['smplx']['transl'])} transls, "
                 f"{len(converted_data['smplx']['global_orient'])} global_orients, {len(converted_data['smplx']['left_hand_pose'])} left_hand_pose, "
                 f"{len(converted_data['smplx']['right_hand_pose'])} right_hand_pose")
        pbar.update()
        pbar.refresh()
    pbar.close()

out['smplx']['body_pose'] = torch.stack(out['smplx']['body_pose'],dim=0)
out['smplx']['left_hand_pose'] = torch.stack(out['smplx']['left_hand_pose'],dim=0)
out['smplx']['right_hand_pose'] = torch.stack(out['smplx']['right_hand_pose'],dim=0)
out['smplx']['global_orient'] = torch.stack(out['smplx']['global_orient'],dim=0)
out['smplx']['betas'] = torch.stack(out['smplx']['betas'],dim=0)
out['smplx']['transl'] = torch.stack(out['smplx']['transl'],dim=0)

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


# ------------------ visualization ----------------
if DO_VIS:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    model_path = f'{dirname(dirname(dirname(abspath(__file__))))}/body_models/'
    smplx_model_female = smplx.create(model_path, model_type='smplx', gender='female', use_pca=False,
                                      flat_hand_mean=True).to(device)
    smplx_model_male = smplx.create(model_path, model_type='smplx', gender='male', use_pca=False,
                                    flat_hand_mean=True).to(device)
    smplx_model_neutral = smplx.create(model_path, model_type='smplx', gender='neutral', use_pca=False,
                                       flat_hand_mean=True).to(device)
    total_cnt_i = 0
    seq_cnt_i = 0
    cnt_i = 0
    seq_i = 7488
    seq = f'/path/to/bedlamcc/png/seq_{seq_i:06d}'
    for frame_i in range(1000):
        print(f'\n\nframe {cnt_i}')
        cnt_i += 1
        img_fn = f'{seq}/seq_{seq_i:06d}_{frame_i:04d}.png'
        if not os.path.exists(img_fn):
            img_fn = f'{seq}/seq_{seq_i:06d}_FinalImage_{frame_i:04d}.png'
        if not os.path.exists(img_fn):
            continue
        #  load images
        im = Image.open(img_fn).convert('RGB')
        # plt.imshow(im); plt.show()
        h, w = im.height, im.width
        gender = out['gender'][frame_i]
        smplx_betas = out['smplx']['betas'][frame_i]
        smplx_body = out['smplx']['body_pose'][frame_i]
        smplx_hand_l = out['smplx']['left_hand_pose'][frame_i]
        smplx_hand_r = out['smplx']['right_hand_pose'][frame_i]
        final_ori = out['smplx']['global_orient'][frame_i]
        final_transl = out['smplx']['transl'][frame_i]
        f = out['focal_length'][frame_i]
        center = out['center'][frame_i]
        scale = out['scale'][frame_i]
        K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]).float().to(device)
        if gender == 'male':
            smplx_model = smplx_model_male
        elif gender == 'female':
            smplx_model = smplx_model_female
        else:
            smplx_model = smplx_model_neutral
        output = smplx_model(betas=smplx_betas, body_pose=smplx_body, left_hand_pose=smplx_hand_l,
                             right_hand_pose=smplx_hand_r,
                             global_orient=final_ori,
                             transl=final_transl,
                             return_verts=True)
        vertices = output.vertices
        pelvis_new = output.joints[0, 0, :, None]
        vert_2D = (K @ (vertices[0].t())).detach()
        vert_2D[0] /= vert_2D[2]
        vert_2D[1] /= vert_2D[2]
        cx, cy = center
        width = height = scale
        top_left_x = cx - width / 2
        top_left_y = cy - height / 2
        _, ax = plt.subplots()
        rect = patches.Rectangle((top_left_x, top_left_y), width, height, linewidth=2, edgecolor='r',
                                 facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)
        ax.plot(cx, cy, "bo")  # Mark the center for reference
        plt.imshow(im);
        plt.scatter(vert_2D[0], vert_2D[1], s=0.2);
        ax.set_xlim(0, w)  # Adjust as needed for your image
        ax.set_ylim(0, h)
        plt.gca().invert_yaxis()
        plt.show()


n_converted_samples = len(out["img_fn"])
print(f'converted to a total of {n_converted_samples} samples')

# import pickle, gzip
# with gzip.open(ann_file_fn, 'wb') as f:
#     pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"saving to {save_fn}")
# np.savez_compressed(save_fn, **out)
torch.save(out, save_fn)

assert len(out['smplx']['transl']) == n_converted_samples, f"ERROR: combined {len(out['smplx']['transl'])} samples, but there should be {n_converted_samples}."
