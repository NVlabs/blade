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

DEV_DIR=/home/rawalk/Desktop/sapiens/seg
DISK_DIR=/home/rawalk/drive/seg

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/home/rawalk/Desktop/sapiens/pose
DISK_DIR=/home/rawalk/drive/pose

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/home/rawalk/Desktop/sapiens/pretrain
DISK_DIR=/home/rawalk/drive/pretrain

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints
