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
import os, os.path as osp, glob, argparse
from api.BLADE_API import BLADE_API


N_IMAGES_PER_BATCH = int(os.environ["MINI_BATCHSIZE"])


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process images with BLADE API.")
    parser.add_argument('image_folder', type=str,
                        help="Path to the folder containing image files (jpg, jpeg, png, etc.).",
                        default='/path/to/image/folder')
    return parser.parse_args()


def main():
    args = parse_args() # Parse command-line arguments
    folder = args.image_folder

    # load images
    image_list = []
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        image_list.extend(glob.glob(osp.join(folder, f'*{ext}')))
    image_list = sorted(image_list)
    if not image_list:
        print(f"No supported image files found in '{folder}'. Please check the folder and file types.")
        sys.exit(0) # Exit gracefully if no images are found
    else:
        print(f"Found {len(image_list)} images in '{folder}'.")
    batch_list = {}
    for idx, img in enumerate(image_list):
        batch_list[str(idx)] = {'rgb_file': img}

    blade = BLADE_API(batch_list=batch_list,
                      render_and_save_imgs=True,
                      temp_output_folder='results/test_demo',
                      device='cuda:0',
                      samples_per_gpu=N_IMAGES_PER_BATCH,
                      workers_per_gpu=8)
    blade.process()


if __name__ == '__main__':
    main()
