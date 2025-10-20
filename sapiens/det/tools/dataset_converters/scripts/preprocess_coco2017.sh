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
#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/val2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/train2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/test2017.zip -d $DATA_ROOT/
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Images/unlabeled2017.zip -d $DATA_ROOT
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/stuff_annotations_trainval2017.zip -d $DATA_ROOT/
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/panoptic_annotations_trainval2017.zip -d $DATA_ROOT/
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/image_info_unlabeled2017.zip -d $DATA_ROOT/
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/image_info_test2017.zip -d $DATA_ROOT/
unzip $DOWNLOAD_DIR/OpenDataLab___COCO_2017/raw/Annotations/annotations_trainval2017.zip -d $DATA_ROOT
rm -rf $DOWNLOAD_DIR/OpenDataLab___COCO_2017
