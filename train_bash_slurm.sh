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


# ======================= NOTE =======================
# set ENV variables MY_CONDA_DIR, TORCH_HOME, WANDB_DIR. the later two avoids storage quota on nodes

# example usage single node:
# export NUM_NODES=1; export NUM_GPUS=8; export MINI_BATCHSIZE=2; export EXP_NAME="depth_${NUM_NODES}x${NUM_GPUS}x${MINI_BATCHSIZE}";
# torchrun --nproc_per_node $NUM_GPUS --master_addr localhost --master_port 12356 --nnodes $NUM_NODES --node_rank 0 ./scripts/train.py ./blade/configs/blade_posenet.py --launcher pytorch --work-dir=./work_dirs/train_depth_${NUM_NODES}x${NUM_GPUS}x${MINI_BATCHSIZE} > log_${NUM_NODES}x${NUM_GPUS}x${MINI_BATCHSIZE}.txt


source ~/.bashrc
: "${MY_CONDA_DIR:?Environment variable MY_CONDA_DIR must be set}"
export miniconda_folder="${MY_CONDA_DIR%/}"
export cur_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
#export TORCH_HOME=/path/to/.cache/
#export WANDB_DIR=/path/to/.wandb_cache/

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
conda_exe="$miniconda_folder/bin/conda"
__conda_setup="$('$conda_exe' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$miniconda_folder/etc/profile.d/conda.sh" ]; then
        . "$miniconda_folder/etc/profile.d/conda.sh"
    else
        export PATH="$miniconda_folder/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate $CUR_CONDA_ENV


#export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=12
echo $CUR_CONDA_ENV
conda env list
conda list

pip install wandb mediapipe  # bug fixed: wandb need to be installed in bash script for some reason, might not be the case for other clusters

export EXP_NAME=$1
export CONFIG_FILE=$2
export WORKDIR="$cur_repo_root/work_dirs/${EXP_NAME}"
echo "SUBMIT_GPUS: ${SUBMIT_GPUS}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "WORKDIR: ${WORKDIR}"
torchrun --nproc_per_node $SUBMIT_GPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes $NUM_NODES --node_rank $NODE_RANK ./scripts/train.py $CONFIG_FILE --launcher pytorch --work-dir="$WORKDIR"
