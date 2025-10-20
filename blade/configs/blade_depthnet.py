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


import os, wandb
from blade.configs.base import (root, body_model_train, test_dict_smplx,
                                body_model_test, wandb_init_args, val_dict,
                                train_depth_withourdata)
find_unused_parameters=True

backbone_half_precision=True
use_syncbn=True

_base_ = ['base.py']
evaluation = dict(metric=['z_error', 'inv_z_error'])
use_adversarial_train = True

checkpoint_config = dict(interval=1, )

optimizer = dict(
    depth_interface=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
    depth_head=dict(type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=210)

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           # dict(type='TensorboardLoggerHook'),
           dict(
               type='WandbLoggerHook',
               init_kwargs=wandb_init_args,
               log_artifact=False,  # Optional: log artifacts like models
               with_step=False,
               by_epoch=True
           ),
           # dict(type='WandbEpochLoggerHook', interval=1, metrics=evaluation['metric'],
           #      init_kwargs=wandb_init_args),
           ])
print("wandb_init_args:")
print(wandb_init_args)
# if 'id' in wandb_init_args:
#     print(wandb_init_args['id'])

checkpoint = None
uv_res = 56
uv_renderer = dict(
    type='UVRenderer',
    resolution=uv_res,
    uv_param_path=f'{root}/body_models/smpl/smpl_uv_decomr.npz',
    bound=(0, 1),
)
depth_renderer = dict(type='depth',
                      resolution=uv_res,
                      blend_params=dict(background_color=(0.0, 0.0, 0.0)))

# model settings
width = 48
downsample = False
use_conv = True

pred_kp3d = True



# sapiens
sapiens_task = 'seg'
sapiens_model_name= 'sapiens_1b'

spapiens_image_size = (768, 1024) ## width x height
sapiens_data_preprocessor = dict(
    type='mmseg.SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, ## convert from bgr to rgb for pretrained models
    pad_val=0,
    size=(spapiens_image_size[1], spapiens_image_size[0]),
    seg_pad_val=255)

if sapiens_model_name == 'sapiens_2b':
    sapiens_embed_dim=1920; sapiens_num_layers=48
elif sapiens_model_name == 'sapiens_1b':
    sapiens_embed_dim = 1536; sapiens_num_layers = 40
else:
    assert False, f"{sapiens_model_name} not implemented"

sapiens_patch_size=16
if sapiens_task == 'depth':
    if sapiens_model_name == 'sapiens_2b':
        sapiens_pretrained_checkpoint=f'{root}/sapiens/checkpoints/depth/sapiens_2b_render_people_epoch_25.pth'
    elif sapiens_model_name == 'sapiens_1b':
        sapiens_pretrained_checkpoint=f'{root}/sapiens/checkpoints/depth/sapiens_1b_render_people_epoch_88.pth'
    else:
        assert False, f"{sapiens_model_name} checkpoint undefined"
elif sapiens_task == 'seg':
    if sapiens_model_name == 'sapiens_1b':
        # 'https://huggingface.co/facebook/sapiens-seg-1b'
        sapiens_pretrained_checkpoint=f'{root}/sapiens/checkpoints/seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth'
    else:
        assert False, f"{sapiens_model_name} checkpoint undefined"
sapiens_norm_cfg = dict(type='SyncBN', requires_grad=True)





model = dict(
    type='BLADE',
    uv_renderer=uv_renderer,
    depth_renderer=depth_renderer,
    depth_backbone_version='vitl',
    depth_head=dict(
        type='DepthOnlyHead',
        in_channels=256,
        znet_config=dict(
            number_of_embed=1,  # 24 + 10
            embed_dim=256,
            nhead=4,
            dim_feedforward=1024,
            numlayers=2,
            max_depth=10,
        ),
    ),
    sapiens_config=dict(
        type='mmseg.DepthEstimator',  # defined in sapiens repo
        data_preprocessor=sapiens_data_preprocessor,
        pretrained=None,
        backbone=dict(
            type='mmpretrain.VisionTransformer',
            arch=sapiens_model_name,
            img_size=(spapiens_image_size[1], spapiens_image_size[0]),
            patch_size=sapiens_patch_size,
            qkv_bias=True,
            # norm_cfg=dict(type='LN'),  # Note: was defaulted to be LN, but mmpretrain's MODELS doesn' have access to MMCV
            final_norm=True,
            drop_path_rate=0.0,
            with_cls_token=False,
            out_type='featmap',
            init_cfg=dict(
                type='Pretrained',
                checkpoint=sapiens_pretrained_checkpoint),
        ),
        decode_head=dict(
            type='mmseg.VitDepthHead',
            in_channels=sapiens_embed_dim,
            channels=384,
            deconv_out_channels=(384, 384, 384, 384),  ## this will 2x at each step. so total is 16x
            deconv_kernel_sizes=(4, 4, 4, 4),
            conv_out_channels=(384, 384),
            conv_kernel_sizes=(1, 1),
            num_classes=1,
            norm_cfg=sapiens_norm_cfg,
            align_corners=False,
            loss_decode=dict(type='SiLogLoss', loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    ),
    body_model_train=body_model_train,
    body_model_test=body_model_test,
    ###
    loss_transl_z=dict(type='L1Loss', loss_weight=1),
    joint_loss_weight=5.,
    vert_loss_weight=5.,
    pretrained_depth_backbone_ckpt=f'{root}/pretrained/model_init_weights/depth_anything_v2_metric_hypersim_vitl.pth',
    depthnet_ckpt_path=None,
    do_res_aug=False,
    miou=True,
    pmiou=True,
    depth_scale=1.,    # a scale factor of 1.2 seems to work better for dfhm
    do_stage_1=True,
    opt_pose=False,      # if True, optimizes pose for improved 2D alignment
    opt_tz=False,
    clear_background=True,    # if True, will detect and segment out human
    convert_to_smpl=False,
    n_optimization_iterations=200,
    render_and_save_imgs=False,
    temp_output_folder=None
)


print(f"MINI BATCHSIZE: {int(os.environ['MINI_BATCHSIZE'])}")
data = dict(samples_per_gpu=int(os.environ['MINI_BATCHSIZE']),
            workers_per_gpu=8,
            train=train_depth_withourdata,
            test=test_dict_smplx,
            val=val_dict,
            )