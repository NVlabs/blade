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
#!/bin/bash

cd ../../.. || exit

SAPIENS_CHECKPOINT_ROOT=/uca/rawalk/sapiens_host
OUTPUT_CHECKPOINT_ROOT=/uca/rawalk/sapiens_lite_host

# MODE='torchscript' ## original. no optimizations (slow). full precision inference.
MODE='bfloat16' ## A100 gpus. faster inference at bfloat16
# MODE='float16' ## V100 gpus. faster inference at float16 (no flash attn)

OUTPUT_CHECKPOINT_ROOT=$OUTPUT_CHECKPOINT_ROOT/$MODE

VALID_GPU_IDS=(0)

#--------------------------MODEL CARD---------------
MODEL_NAME='sapiens_1b'; CHECKPOINT=$SAPIENS_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_goliath_coco_wholebody_mpii_crowdpose_aic_best_goliath_AP_640.pth

DATASET='goliath_coco_wholebody_mpii_crowdpose_aic'
MODEL="${MODEL_NAME}-210e_${DATASET}-1024x768"
POSE_CONFIG_FILE="configs/sapiens_pose/${DATASET}/${MODEL}.py"

OUTPUT=${OUTPUT_CHECKPOINT_ROOT}/pose/checkpoints/${MODEL_NAME}/

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=${VALID_GPU_IDS[0]} python3 tools/deployment/torch_optimization.py \
            ${POSE_CONFIG_FILE} ${CHECKPOINT} --output-dir ${OUTPUT} --explain-verbose
