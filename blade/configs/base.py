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


import socket, os, sys
project_name = 'BLADE'
exp_name = os.environ.get('EXP_NAME', 'none')
evaluation = dict(metric=['pa-mpjpe', 'mpjpe', 'pve', 'pa-pve', 'miou', 'pmiou', 'z_error'])

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
repo_folder_name = root.split('/')[-1]

# get wandb id
custom_imports = dict(
    imports=['blade.custom_hooks.wandb_hook'],  # dotted path to the file
    allow_failed_imports=False)
wandb_init_args = {
    'entity': '',
    'project': project_name,
    'name': exp_name,  # Optional: specify your W&B entity
    'resume': 'allow'
}


convention = 'smpl_54'
convention_test = 'h36m'  # 3dpw
# convention_test = 'smpl'  # pdhuman-syn-test, spec-mtp

# img_res = 224
img_res = 518
img_res_train = 518
# img_res_posenet = 1024
img_res_posenet = 800

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

checkpoint_config = dict(interval=20, )

data_keys = [
    'has_smpl', 'has_transl', 'has_focal_length', 'has_keypoints2d',
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d',
    # 'has_keypoints3d', 'keypoints3d',  # note: removed because not accurate, get 3d kpts from SMLP directly
    'sample_idx', 'has_K', 'K',
    'is_flipped',
    # 'ori_keypoints2d',
    'smpl_origin_orient', 'ori_shape',
    'center', 'scale', 'bbox_info', 'ori_focal_length', 'inv_trans', 'trans',
    'img_h', 'img_w', 'distortion_max', 'is_distorted', 'rotation'
]

data_keys_test = [
    'has_smpl', 'has_transl', 'has_focal_length', 'smpl_body_pose',
    'smpl_global_orient', 'smpl_betas', 'smpl_transl',
    'keypoints2d',
    # 'has_keypoints3d', 'keypoints3d',  # note: removed because not accurate, get 3d kpts from SMLP directly
    'sample_idx', 'has_K', 'K',
    # 'ori_keypoints2d',
    'ori_shape', 'center', 'scale', 'bbox_info',
    'ori_focal_length', 'img_h', 'img_w'
]

data_keys_inference = [
    'ori_shape', 'center', 'scale', 'bbox_info', 'ori_focal_length'
]

data_keys_adv = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_pipeline_adv = [dict(type='Collect', keys=data_keys_adv, meta_keys=[])]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         # rot_factor=30,
         # scale_factor=0.35,
         # rot_prob=0.5,
         # scale_add=0.1
         rot_factor=0,
         scale_factor=0.,
         rot_prob=0.,
         scale_add=0.
         ),
    dict(type='GetBboxInfo',
         # rand_shift=0.2,
         # rand_shift_prob=1.
        rand_shift=0.,
         rand_shift_prob=0.
         ),
    dict(type='MeshAffine',
         img_res=dict(img=518),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

train_pipeline_noaug = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         rot_factor=0,
         scale_factor=0.,
         rot_prob=0.,
         scale_add=0.
         ),
    dict(type='GetBboxInfo',
        rand_shift=0.,
         rand_shift_prob=0.
         ),
    dict(type='MeshAffine',
         img_res=dict(img=518),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]


train_pipeline_lessaug = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         rot_factor=30,
         scale_factor=0.15,
         rot_prob=0.5,
         scale_add=0.),
    dict(type='GetBboxInfo',
         rand_shift=0.1,
         rand_shift_prob=1.),
    dict(type='MeshAffine',
         img_res=dict(img=img_res_train),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

train_pipeline_smplx = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
        rot_factor=270,
         scale_factor=0.,
         rot_prob=0.8,
         scale_add=0.
         ),
    dict(type='GetBboxInfo',
         rand_shift=0.,
         rand_shift_prob=0.),
    dict(type='MeshAffine',
         img_res=dict(img=img_res_train),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

train_pipeline_smplx_pose = [
        dict(type='LoadImageFromFile'),

        # depthnet loader, no augmentation
        dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=0., scale_factor=0., rot_prob=0., scale_add=0.),
        dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
        dict(type='MeshAffine', img_res=dict(img=img_res_train), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation']),

        # posenet loader, augmented
        dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0.5, convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=270, scale_factor=0., rot_prob=0.8, scale_add=0. ),
        dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
        dict(type='MeshAffine', img_res=dict(img=img_res_posenet), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation'])
        ]


train_pipeline_smplx_pose_lessaug = [
        dict(type='LoadImageFromFile'),

        # depthnet loader, no augmentation
        dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=0., scale_factor=0., rot_prob=0., scale_add=0.),
        dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
        dict(type='MeshAffine', img_res=dict(img=img_res_train), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation']),

        # posenet loader, augmented
        dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0.5, convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0., rot_prob=0.8, scale_add=0. ),
        dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
        dict(type='MeshAffine', img_res=dict(img=img_res_posenet), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation'])
        ]

train_pipeline_smplx_pose_lessaug_bedlamcc = [
        dict(type='LoadImageFromFile'),

        # depthnet loader, no augmentation
        dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=0., scale_factor=0., rot_prob=0., scale_add=0.),
        dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
        dict(type='MeshAffine', img_res=dict(img=img_res_train), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation']),

        # posenet loader, augmented
        dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
        dict(type='RandomHorizontalFlip', flip_prob=0.5, convention=convention, img_fields=['img']),
        dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25, rot_prob=0.8, scale_add=0.25 ),
        dict(type='GetBboxInfo', rand_shift=0.15, rand_shift_prob=0.3),
        dict(type='MeshAffine', img_res=dict(img=img_res_posenet), require_origin_kp2d=True, img_fields=['img']),
        dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=data_keys),
        dict(type='Collect',
             keys=['img', *data_keys],
             meta_keys=['image_path', 'center', 'scale', 'rotation'])
        ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline_depth = [
    dict(type='LoadImageFromFile'),

    # depthnet loader, crops to center square
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation']),

    # posenet loader, not cropped
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res_posenet),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline_smplx = [
    dict(type='LoadImageFromFile'),

    # depthnet loader, crops to center square
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation']),

    # posenet loader, not cropped
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    # dict(type='MeshAffine', img_res=img_res_posenet),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]


test_pipeline_smplx_ourdata = [
    dict(type='LoadImageFromFile'),

    # depthnet loader, no augmentation
    dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
    dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
    dict(type='GetRandomScaleRotation', rot_factor=0., scale_factor=0., rot_prob=0., scale_add=0.),
    dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
    dict(type='MeshAffine', img_res=dict(img=img_res_train), require_origin_kp2d=True, img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation']),

    # posenet loader, augmented
    dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
    dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0., rot_prob=0., scale_add=0.),
    dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
    dict(type='MeshAffine', img_res=dict(img=img_res_posenet), require_origin_kp2d=True, img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]


inference_pipeline = [
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0, rot_prob=0),

    # depthnet
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation']),

    # posenet
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res_posenet),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline_batchof1 = [
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0, rot_prob=0),

    # depthnet
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation']),

    # posenet
    dict(type='GetBboxInfo'),
    # dict(type='MeshAffine', img_res=img_res_posenet),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]


cache_files = {
    'h36m': f'{root}/cache/h36m_train_{convention}.npz',
    'h36m_mosh': f'{root}/cache/h36m_mosh_train_{convention}.npz',
    'h36m_transl': f'{root}/cache/h36m_mosh_train_transl_{convention}.npz',
    'mpi_inf_3dhp': f'{root}/cache/mpi_inf_3dhp_train_{convention}.npz',
    'lsp': f'{root}/cache/lsp_train_{convention}.npz',
    'lspet_eft': f'{root}/cache/lspet_eft_{convention}.npz',
    'mpii': f'{root}/cache/mpii_train_{convention}.npz',
    'mpii_cliff': f'{root}/cache/mpii_cliff_{convention}.npz',
    'muco': f'{root}/cache/muco_{convention}.npz',
    'coco2014': f'{root}/cache/coco_2014_train_{convention}.npz',
    'coco_cliff': f'{root}/cache/coco_cliff_train_{convention}.npz',
    'coco2017': f'{root}/cache/coco_2017_train_{convention}.npz',
    'spec_train': f'{root}/cache/spec_train_{convention}.npz',
    'agora_train': f'{root}/cache/agora_train_{convention}.npz',
    'agora_transl': f'{root}/cache/agora_train_transl_{convention}.npz',
    'agora_val_transl': f'{root}/cache/agora_val_transl_{convention}.npz',
    'pw3d_train': f'{root}/cache/pw3d_train_{convention}.npz',
    'pw3d_transl': f'{root}/cache/pw3d_train_transl_{convention}.npz',
    'pdhuman_train': f'{root}/cache/pdhuman_train_{convention}.npz',
    'pdhuman_train2': f'{root}/cache/pdhuman_train2_{convention}.npz',
    'humman_train': f'{root}/cache/humman_train_{convention}.npz',
    'humman_train2': f'{root}/cache/humman_train2_{convention}.npz',
    'spec_mtp': f'{root}/cache/spec_mtp_{convention}.npz',
    'bedlamcc_train': f'{root}/cache/bedlamcc.npz',
    'bedlamcc_eval': f'{root}/cache/bedlamcc_eval.npz',
    'h36m_transl_smplx': f'{root}/cache/h36m_mosh_train_transl_smplx.npz',
    'humman_train_smplx': f'{root}/cache/humman_train_smplx.npz',
    'pdhuman_train_smplx': f'{root}/cache/pdhuman_train_smplx.npz',
}

body_model_3dpw = dict(
    type='GenderedSMPL',
    keypoint_src='h36m',
    keypoint_dst='h36m',
    model_path=f'{root}/body_models/smpl',
    joints_regressor=f'{root}/body_models/smpl/J_regressor_h36m.npy')

body_model_test = dict(
    type='SMPL',
    keypoint_src='h36m',
    keypoint_dst='h36m',
    model_path=f'{root}/body_models/smpl',
    joints_regressor=f'{root}/body_models/smpl/J_regressor_h36m.npy')

body_model_train = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst=convention,
    model_path=f'{root}/body_models/smpl',
    keypoint_approximate=True,
    extra_joints_regressor=f'{root}/body_models/smpl/J_regressor_extra.npy')

humman_test_p3 = dict(type='HumanImageDataset',
                      body_model=body_model_test,
                      dataset_name='humman',
                      is_distorted=True,
                      test_mode=True,
                      data_prefix=f'{root}/mmhuman_data/',
                      pipeline=test_pipeline,
                      ann_file='humman_test_p3.npz')

pdhuman_test = dict(type='HumanImageDataset',
                    data_prefix=f'{root}/mmhuman_data/',
                    convention=convention_test,
                    ann_file='pdhuman_test.npz',
                    dataset_name='pdhuman',
                    test_mode=True,
                    is_distorted=True,
                    body_model=body_model_test,
                    pipeline=test_pipeline)
pdhuman_test_p5 = pdhuman_test.copy()
pdhuman_test_p5['ann_file'] = 'pdhuman_test_p5.npz'

spec_mtp = dict(
    type='HumanImageDataset',
    data_prefix=f'{root}/mmhuman_data/',
    convention=convention,
    ann_file='spec_mtp.npz',
    dataset_name='spec_mtp',
    test_mode=True,
    body_model=body_model_test,
    pipeline=test_pipeline,
)

spec_mtp_p1 = spec_mtp.copy()
spec_mtp_p1['ann_file'] = 'spec_mtp_p1.npz'
spec_mtp_p2 = spec_mtp.copy()
spec_mtp_p2['ann_file'] = 'spec_mtp_p2.npz'
spec_mtp_p3 = spec_mtp.copy()
spec_mtp_p3['ann_file'] = 'spec_mtp_p3.npz'


pdhuman_train_smplx = dict(type='HumanImageDataset_SMPLX',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train_smplx.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     cache_data_path=cache_files['pdhuman_train_smplx'],
                     pipeline=train_pipeline_smplx_pose)

pdhuman_train_smplx_lessaug = dict(type='HumanImageDataset_SMPLX',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train_smplx.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     do_black_pad_aug=False,
                     cache_data_path=cache_files['pdhuman_train_smplx'],
                     pipeline=train_pipeline_smplx_pose_lessaug)

pdhuman_train_tz = dict(type='HumanImageDataset_Tz',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     cache_data_path=cache_files['pdhuman_train'],
                     pipeline=train_pipeline)

pdhuman_train_tz_noaug = dict(type='HumanImageDataset_Tz',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     cache_data_path=cache_files['pdhuman_train'],
                     pipeline=train_pipeline_noaug)

pdhuman_train_tz_lessaug = dict(type='HumanImageDataset_Tz',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     cache_data_path=cache_files['pdhuman_train'],
                     pipeline=train_pipeline_lessaug)

bedlamcc_train = dict(type='OurDataset',
                   data_prefix=f'{root}/mmhuman_data/',
                   convention=convention,
                   ann_file='bedlamcc.npz',
                   dataset_name='bedlamcc',
                   is_distorted=False,
                   cache_data_path=cache_files['bedlamcc_train'],
                   pipeline=train_pipeline)

bedlamcc_eval = dict(type='OurDataset_SMPLX',
                   data_prefix=f'{root}/mmhuman_data/',
                   convention=convention,
                   dataset_name='bedlamcc_eval',
                   data_folder='bedlamcc_eval',
                   ann_file='bedlamcc_eval.npz',
                   cache_data_path=cache_files['bedlamcc_eval'],
                   is_distorted=False,
                   is_test=True,
                   # do_center_crop=True,
                   pipeline=test_pipeline_smplx_ourdata)

h36m_mosh_transl_smplx = dict(type='HumanImageDataset_SMPLX',
                        convention=convention,
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline_smplx_pose,
                        cache_data_path=cache_files['h36m_transl_smplx'],
                        ann_file='h36m_mosh_train_transl_smplx.npz',)

h36m_mosh_transl_smplx_lessaug = dict(type='HumanImageDataset_SMPLX',
                        convention=convention,
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline_smplx_pose_lessaug,
                        do_black_pad_aug=False,
                        cache_data_path=cache_files['h36m_transl_smplx'],
                        ann_file='h36m_mosh_train_transl_smplx.npz',)

h36m_mosh_transl_tz = dict(type='HumanImageDataset_Tz',
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline,
                        convention=convention,
                        cache_data_path=cache_files['h36m_transl'],
                        ann_file='h36m_mosh_train_transl.npz')

h36m_mosh_transl_tz_noaug = dict(type='HumanImageDataset_Tz',
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline_noaug,
                        convention=convention,
                        cache_data_path=cache_files['h36m_transl'],
                        ann_file='h36m_mosh_train_transl.npz')

h36m_mosh_transl_tz_lessaug = dict(type='HumanImageDataset_Tz',
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline_lessaug,
                        convention=convention,
                        cache_data_path=cache_files['h36m_transl'],
                        ann_file='h36m_mosh_train_transl.npz')

pdhuman_test_smplx = dict(type='HumanImageDataset_SMPLX',
                    data_prefix=f'{root}/mmhuman_data/',
                    convention=convention_test,
                    ann_file='pdhuman_test.npz',
                    dataset_name='pdhuman',
                    test_mode=True,
                    is_distorted=True,
                    body_model=body_model_test,
                    pipeline=test_pipeline_smplx,
                    # num_data=10,
                    )
pdhuman_test_p5_smplx = pdhuman_test_smplx.copy()
pdhuman_test_p5_smplx['ann_file'] = 'pdhuman_test_p5.npz'


spec_mtp_smplx = dict(
    type='HumanImageDataset_SMPLX',
    data_prefix=f'{root}/mmhuman_data/',
    convention=convention,
    ann_file='spec_mtp.npz',
    dataset_name='spec_mtp',
    test_mode=True,
    body_model=body_model_test,
    pipeline=test_pipeline_smplx,
    # num_data=5,
)
spec_mtp_p3_smplx = spec_mtp_smplx.copy()
spec_mtp_p3_smplx['ann_file'] = 'spec_mtp_p3.npz'

humman_train_smplx = dict(type='HumanImageDataset_SMPLX',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline_smplx_pose,
                    cache_data_path=cache_files['humman_train_smplx'],
                    ann_file='humman_train_smplx.npz')

humman_train_smplx_lessaug = dict(type='HumanImageDataset_SMPLX',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline_smplx_pose_lessaug,
                    do_black_pad_aug=False,
                    cache_data_path=cache_files['humman_train_smplx'],
                    ann_file='humman_train_smplx.npz')

humman_train_tz = dict(type='HumanImageDataset_Tz',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['humman_train'],
                    ann_file='humman_train.npz')

humman_train_tz_noaug = dict(type='HumanImageDataset_Tz',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline_noaug,
                    cache_data_path=cache_files['humman_train'],
                    ann_file='humman_train.npz')

humman_train_tz_lessaug = dict(type='HumanImageDataset_Tz',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline_lessaug,
                    cache_data_path=cache_files['humman_train'],
                    ann_file='humman_train.npz')

humman_test_p3_smplx = dict(type='HumanImageDataset_SMPLX',
                      body_model=body_model_test,
                      dataset_name='humman',
                      is_distorted=True,
                      test_mode=True,
                      data_prefix=f'{root}/mmhuman_data/',
                      pipeline=test_pipeline_smplx,
                      ann_file='humman_test_p3.npz')

train_test = dict(type='MixedDataset',
                         configs=[
                             humman_train_tz
                         ],
                         partition=[1.])

train_depth_old3 = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz
                               ],
                               partition=[0.29, 0.28, 0.43]
                        )

train_depth_old3_noaug = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz_noaug, humman_train_tz_noaug, h36m_mosh_transl_tz_noaug
                               ],
                               partition=[0.29, 0.28, 0.43]
                        )

train_depth_withourdata = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz, bedlamcc_train
                               ],
                               partition=[0.145, 0.140, 0.215, 0.5]
                        )

train_depth_withourdata_lesspdhuman = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz, bedlamcc_train
                               ],
                               partition=[0.105, 0.140, 0.255, 0.5]
                        )

train_depth_withourdata_morepdhuman = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz, bedlamcc_train
                               ],
                               partition=[0.285, 0.050, 0.165, 0.5]
                        )

train_depth_withourdata_moreours = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz, bedlamcc_train
                               ],
                               partition=[0.087, 0.084, 0.129, 0.7]
                        )

train_depth_withourdata_lessours = dict(type='MixedDataset',
                               configs=[
                                   pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz, bedlamcc_train
                               ],
                               partition=[0.1885, 0.182, 0.2795, 0.35]
                        )

train_depth_existingdata = dict(type='MixedDataset',
                                configs=[
                                    pdhuman_train_tz, humman_train_tz, h36m_mosh_transl_tz
                                ],
                                partition=[0.29, 0.28, 0.430]
                                )

train_smplx_lessaug_morehummanpdhuman = dict(type='MixedDataset',
                         configs=[
                             pdhuman_train_smplx_lessaug, humman_train_smplx_lessaug, h36m_mosh_transl_smplx_lessaug
                         ],
                         partition=[0.435, 0.436, 0.129])

itw_dataset = dict(type='ITWDataset',
                               pipeline=inference_pipeline,
                               batch_list=[])

test_dict_smplx = dict(
    humman_p3=humman_test_p3_smplx,  # 1.8
    pdhuman_p5=pdhuman_test_p5_smplx,  # 3.0
    spec_mtp_p3=spec_mtp_p3_smplx,    # 1.8
    bedlamcc_eval=bedlamcc_eval,
    )

val_dict = spec_mtp_p3_smplx

