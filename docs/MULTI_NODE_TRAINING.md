# Training Guide

## Multi-Node Training on Slurm Clusters

This guide provides detailed instructions for training BLADE on multi-node Slurm-like clusters.

### Environment Setup

### Two-Stage Training Process

Set environment variables shared by both stages:
```bash
export REPO=<repo_name>                            # your repository name
export REPO_PARENT=<absolute_path_to_repo_parent>  # absolute path to repo's parent directory
export MY_CONDA_DIR=<path_to_miniconda_root>       # path to your miniconda root folder
export TORCH_HOME=<path to ./cache folder>           # path to your .cache/ folder
export WANDB_DIR=<path to ./wandb_cache folder>             # path to your .wandb_cache/ folder
export CUR_CONDA_ENV=blade_env                     # conda environment name
```

#### Stage 1: DepthNet Training

```bash
export CONFIG=blade_depthnet.py                    # or blade_posenet.py
export VERSION=${REPO}_depthnet                    # or ${REPO}_posenet for job naming
export NUM_NODES=8                                 # number of nodes
export SUBMIT_GPUS=8                               # GPUs per node
export MINI_BATCHSIZE=2                            # mini batch size

# Launch DepthNet training job
launch_job --name ${VERSION}_${NUM_NODES}x${SUBMIT_GPUS}_mbs${MINI_BATCHSIZE} 
        --gpu $SUBMIT_GPUS --nodes $NUM_NODES 
        --command 'export CUR_CONDA_ENV=${CUR_CONDA_ENV}; export REPO=${REPO}; 
                  export REPO_PARENT=${REPO_PARENT}; export MY_CONDA_DIR=${MY_CONDA_DIR}; 
                  export TORCH_HOME=${TORCH_HOME}; export WANDB_DIR=${WANDB_DIR}; 
                  cd ${REPO_PARENT}/${REPO}/;  export MINI_BATCHSIZE=${MINI_BATCHSIZE}; 
                  bash train_bash_slurm.sh ${VERSION}_${NUM_NODES}x${SUBMIT_GPUS}_mbs${MINI_BATCHSIZE} 
                  ./blade/configs/${CONFIG}'
```

> **Expected Duration**: Best performance often achieved within 4 epochs.
> 
> **Note**: 8 GPUs with batch size 16 works, we used 8 nodes just for training speed


#### Stage 2: PoseNet Training

```bash
export CONFIG=blade_posenet.py                    # or blade_posenet.py
export VERSION=${REPO}_posenet                    # or ${REPO}_posenet for job naming
export NUM_NODES=6                                 # number of nodes
export SUBMIT_GPUS=8                               # GPUs per node
export MINI_BATCHSIZE=7                            # mini batch size


###### important!!!! #########
# 1. Set 'depthnet_ckpt_path' in blade_posenet.py to the Stage 1 DepthNet checkpoint path


# Launch PoseNet training job
<launch job e.g. sbatch> -n ${VERSION}_${NUM_NODES}x${SUBMIT_GPUS}_mbs${MINI_BATCHSIZE} \
         --gpu $SUBMIT_GPUS --nodes $NUM_NODES --image <your_instance_snapshot> \
         --command 'export MY_CONDA_DIR=${MY_CONDA_DIR}; export TORCH_HOME=${TORCH_HOME}; export WANDB_DIR=${WANDB_DIR}; export CUR_CONDA_ENV=${CUR_CONDA_ENV}; export REPO=${REPO}; cd ${REPO_PARENT}/${REPO}/; export MINI_BATCHSIZE=${MINI_BATCHSIZE}; bash train_bash_slurm.sh ${VERSION}_${NUM_NODES}x${SUBMIT_GPUS}_mbs${MINI_BATCHSIZE} ./blade/configs/${CONFIG} ${DEPTH_CKPT}'
```

> **Expected Duration**: Usually converges around epoch 1-2

### Training Scripts

The training process uses the following scripts:

- **`train_bash_slurm.sh`**: Training launcher script
- **`scripts/train.py`**: Core training script
- **`blade/configs/blade_depthnet.py`**: DepthNet configuration
- **`blade/configs/blade_posenet.py`**: PoseNet configuration

### Configuration Files

#### Main Configuration Files
- **Main config**: `<blade root>/blade/configs/blade_posenet.py`
- **DepthNet config**: `<blade root>/blade/configs/blade_depthnet.py`
- **PoseNet config**: `<blade root>/blade/configs/blade_posenet.py`
- **API config**: `<blade root>/api/BLADE_API.py`

### Monitoring Training

Weights & Biases (WandB) entity/project/experiment name defined in `blade/configs/base.py` 

#### Log Files
Check `<blade root>/work_dirs/<experiment_name>/`

### Dataset Requirements

Ensure your datasets are properly set up:

```bash
# Create symlinks to datasets
cd <repo_root>
ln -s <path_to_dataset_root>/datasets mmhuman_data/datasets
ln -s <path_to_dataset_root>/preprocessed_datasets mmhuman_data/preprocessed_datasets
```

### Supported Datasets

- **HuMMan**: [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
- **H36M**: [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) + [ZOLLY](https://github.com/SMPLCap/Zolly)
- **PDHuman**: [ZOLLY](https://github.com/SMPLCap/Zolly)
- **BEDLAM-CC**: Custom rendering (see [DATASETS.md](DATASETS.md))

### Troubleshooting

### Checkpoint Usage

#### Saving Checkpoints
- Checkpoints are automatically saved in `work_dirs/`

#### Resuming Training
```bash
# use the --resume-from flag
python scripts/train.py ... --resume-from <path_to_checkpoint>
```

### Validation and Testing
#### Test with 1 GPU
```bash
# Test trained model
python scripts/test.py ./blade/configs/blade_posenet.py \
--work-dir=<working directory>
--out <path to save to like output.pth>
--data-name <spec_mtp_p3, pdhuman_p5, or humman_p3>
--checkpoint <path to checkpoint>
```

#### Test with multiple GPUs
```bash
export CUR_CONDA_ENV=blade_env;
conda deactivate; conda activate $CUR_CONDA_ENV; 
export MASTER_ADDR=localhost; export MASTER_PORT=12355; export NODE_RANK=0;
export NUM_NODES=1;      export SUBMIT_GPUS=8;     export MINI_BATCHSIZE=16;
 
export CONFIG=blade_posenet.py;                            # or blade_posenet.py
export VERSION=test_blade;   # or ${REPO}_posenet, for job name or logging
export MY_CONDA_DIR=</path/to/miniconda3/>;

# if DISK QUOTA ERROR on SLURM due to home dir size
export TORCH_HOME=</path/to/.cache/>; export WANDB_DIR=</path/to/.wandb_cache/>; 

export TESTSET=<spec_mtp_p3 or pdhuman_p5 or humman_p3>;
export CKPT=<path to checkpoint .pth> # e.g. pretrained/epoch_2.pth
bash test_bash_slurm.sh ${VERSION}_${NUM_NODES}x${SUBMIT_GPUS}_mbs${MINI_BATCHSIZE} \
      ./blade/configs/${CONFIG}
```

### Resource Requirements
**Minimum Requirements:** 8x GPUs, 24GB per GPU

### Trouble Shooting
- Fix EGL issues if needed:
```bash
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
[ -f "$CONDA_PREFIX/lib/libEGL.so" ] || ln -s "$CONDA_PREFIX/lib/libEGL.so.1" "$CONDA_PREFIX/lib/libEGL.so"
```
