# Dataset Guide

This guide covers dataset setup, preprocessing, and conversion for BLADE training.

## Dataset Structure

### Required Directory Structure
```
<BLADE_repo_root>/mmhuman_data/
├── datasets/                    # Raw dataset images
│   ├── humman/
│   │   ├── test_images/
│   │   └── train_images/
│   ├── h36m/
│   │   ├── S1/ S11/ S5/ S6/ S7/ S8/ S9/
│   ├── pdhuman/
│   │   └── imgs/
│   └── spec_mtp/
│       └── imgs/
└── preprocessed_datasets/       # Processed labels
    ├── # Testing (original SMPL labels)
    ├── spec_mtp_p3.npz
    ├── pdhuman_test_p5.npz
    ├── humman_test_p3.npz
    ├── # Training (converted to SMPL-X)
    ├── pdhuman_train_smplx.npz
    ├── humman_train_smplx.npz
    ├── h36m_mosh_train_transl_smplx.npz
    └── # Original SMPL labels for conversion
        ├── pdhuman_train.npz
        ├── humman_train.npz
        └── h36m_mosh_train_transl.npz
```

## Dataset Setup

### 1. Create Symlinks
```bash
cd <repo_root>
ln -s <path_to_dataset_root>/datasets mmhuman_data/datasets
ln -s <path_to_dataset_root>/preprocessed_datasets mmhuman_data/preprocessed_datasets
```

### 2. Download Datasets

#### HuMMan Dataset
```bash
# Follow MMHuman3D preprocessing guide
# https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md
```

#### H36M Dataset
```bash
# Download from MMHuman3D
# https://github.com/open-mmlab/mmhuman3d/blob/main/docs/preprocess_dataset.md

# Additional H36M MoSH + translation data from ZOLLY
# Contact: https://github.com/SMPLCap/Zolly
```

#### PDHuman Dataset
```bash
# Download from ZOLLY
# https://github.com/SMPLCap/Zolly
```

## Dataset Conversion

### Converting SMPL Datasets to SMPL-X

Due to licensing restrictions, SMPL-X converted labels must be generated locally.

#### Single Node Conversion
```bash
# Convert datasets (e.g., pdhuman, h36m, humman)
python blade/datasets/combine_preprocessed_smpldata.py <dataset_name>
```

#### Multi-Node Conversion (Slurm Clusters)

For large-scale conversion on CPU nodes:

```bash
# Set environment variables
export ENV_NAME=blade_env
export DATASET=pdhuman                    # or 'h36m', 'humman'
export MACHINE_RANK=0                     # change for each node
export NUM_PROCESSES_PER_MACHINE=24       # processes per machine
export IMAGES_PER_PROCESS=439             # batch size per process
export REPO_ABS_PATH=<absolute_path_to_repo>

# Launch conversion job
<launch job e.g. sbatch> -n preprocess_${DATASET}_${MACHINE_RANK} \
         --cpu $NUM_PROCESSES_PER_MACHINE --image <container_image> \
         --command 'cd ${REPO_ABS_PATH}/blade/datasets/; bash batch_run_preprocess_smpl2smplx.sh ${ENV_NAME} ${DATASET} ${MACHINE_RANK} ${NUM_PROCESSES_PER_MACHINE} ${IMAGES_PER_PROCESS}'
```

#### Conversion Process Details

1. **Parallel Processing**: Launch on multiple CPU nodes for faster conversion
2. **Environment Variables**:
   - `MACHINE_RANK`: Node rank (0, 1, 2, ...)
   - `IMAGES_PER_PROCESS`: Number of images per process
   - `NUM_PROCESSES_PER_MACHINE`: Number of processes per machine
   - Total number of images = `IMAGES_PER_PROCESS` × `NUM_PROCESSES_PER_MACHINE`

3. **Output**: Files stored in `mmhuman_data/preprocessed_datasets/*_smplx_chunks/`
   - Format: `*_smplx_{PROCESS_ID}_{IMG_START}_{IMG_END}.npz`
   - Example: `h36m_mosh_train_transl_smplx_0_0_438.npz`

4. **Combination**: After all conversions complete:
   ```bash
   cd <repo>
   python blade/datasets/combine_preprocessed_smpldata.py <dataset_name>
   ```

## BEDLAM-CC Dataset

### Overview
BEDLAM-CC (BEDLAM-Close-Camera) is based on the [BEDLAM](https://bedlam.is.tue.mpg.de/index.html) dataset but focuses on close-camera scenarios.
>Due to license restrictions, we cannot share the rendered images and thus provide a guide to generating a dataset similar to ours. 

### Rendering Process
Please refer to [BEDLAMCC_GENERATION](docs/BEDLAMCC_GENERATION.md)

### Preprocessing BEDLAM-CC

#### Single Node Processing
```bash
# Preprocess BEDLAM-CC data
python blade/datasets/combine_preprocessed_ourdata.py
```

#### Multi-Node Processing (Slurm)
```bash
# Set environment variables
export ENV_NAME=blade_env
export MACHINE_RANK=0
export NUM_PROCESSES_PER_MACHINE=24
export SEQ_PER_PROCESS=3                    # sequences per process
export REPO_ABS_PATH=<absolute_path_to_repo>

# Launch preprocessing job
<launch job e.g. sbatch> -n preprocess_${MACHINE_RANK} \
         --cpu $NUM_PROCESSES_PER_MACHINE --image <container_image> \
         --command 'cd ${REPO_ABS_PATH}/blade/datasets/; bash batch_run_preprocess_ourdata.sh ${ENV_NAME} ${MACHINE_RANK} ${NUM_PROCESSES_PER_MACHINE} ${SEQ_PER_PROCESS}'
```

#### BEDLAM-CC Processing Details

1. **Sequences**: Each sequence contains hundreds of random views of a dynamic person
2. **Environment Variables**:
   - `MACHINE_RANK`: Node rank
   - `SEQ_PER_PROCESS`: Sequences per process
   - `NUM_PROCESSES_PER_MACHINE`: Processes per machine
   - Total sequences = `SEQ_PER_PROCESS` × `NUM_PROCESSES_PER_MACHINE`

3. **Output**: Files stored in `mmhuman_data/preprocessed_datasets/bedlamcc_smplx_chunks/`
   - Format: `bedlamcc_smplx_{PROCESS_ID}_{IMG_START}_{IMG_END}.npz`

4. **Combination**: After all processing complete:
   ```bash
   cd <BLADE_repo_root>
   python blade/datasets/combine_preprocessed_ourdata.py
   ```

## Dataset Configuration

### Dataset Classes
- **HumanImageDataset_Tz**: Main dataset class for Stage 1 DepthNet training
- **HumanImageDataset_SMPLX**: Main dataset class for Stage 2 PoseNet training
- **BedlamDataset**: Custom dataset class for BEDLAM-CC

## Data Preprocessing Scripts

### Available Scripts

1. **`batch_run_preprocess_smpl2smplx.sh`**: SMPL to SMPL-X conversion of existing datasets like SPEC, HuMMan, H36M, and PDHuman
2. **`batch_run_preprocess_ourdata.sh`**: process raw BEDLAM-CC into chunks with camera, pose, and gender info
3. **`combine_preprocessed_smpldata.py`**: Combine converted chunks of a datasets into a single .npz file
4. **`combine_preprocessed_ourdata.py`**: Combine BEDLAM-CC chunks into a single .npz file

### Script Parameters

#### SMPL to SMPL-X Conversion
- `ENV_NAME`: Conda environment name
- `DATASET`: Dataset name (pdhuman, h36m, humman)
- `MACHINE_RANK`: Node rank
- `NUM_PROCESSES_PER_MACHINE`: Processes per machine
- `IMAGES_PER_PROCESS`: Images per process

#### BEDLAM-CC Preprocessing
- `ENV_NAME`: Conda environment name
- `MACHINE_RANK`: Node rank
- `NUM_PROCESSES_PER_MACHINE`: Processes per machine
- `SEQ_PER_PROCESS`: Sequences per process (each sequence has many images)

### Performance Tips

1. **SMPL & SMPL-X Conversion Speed**: Make sure the number of cores equals to the number of processes you are running. 
Too many cores accutally leads to slower processing for some reason. Similarly multiprocessing in Python is extremely slow.