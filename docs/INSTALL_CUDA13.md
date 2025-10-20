# Installation Guide

## CUDA 11.8 Installation

Follow [README.md](../README.md).

## CUDA 13 Installation

For CUDA other than 11.8, e.g. 13, you likely need to install CUDA 11.8 first.

### Prerequisites
- CUDA 13+ installed
- Python 3.9.19
- Conda
- GCC 11.x

### Step 1: CUDA 11.8 and GCC 11.x Setup
```bash
# Install CUDA 11.8 toolkit
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=$CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Verify installation
test -f "$CUDA_HOME/include/cusparse.h" && echo cusparse.h OK
test -f "$CUDA_HOME/include/thrust/complex.h" && echo thrust OK
which nvcc && nvcc --version
```

### Step 2: Install GCC 11.x and CUDA Toolkit
```bash
# Install GCC 11.x and CUDA toolkit via conda
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
conda install -y -c nvidia cuda-toolkit=11.8.0 cuda-nvcc=11.8.89 cuda-cudart-dev=11.8.89
conda install -y -c nvidia cuda-cusparse-dev=11.8.0 cuda-cusparse=11.8.0

# Set compiler environment
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCCFLAGS="--compiler-bindir=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# Verify GCC version
$CC --version | head -1
$CXX --version | head -1
```

### Step 3: Install Thrust & CUB
```bash
# Clone and install Thrust
cd /tmp
git clone --recursive --branch 1.16.0 https://github.com/NVIDIA/thrust.git thrust-1.16
mkdir -p "$CONDA_PREFIX/include"
rsync -a thrust-1.16/thrust "$CONDA_PREFIX/include/"
rsync -a thrust-1.16/dependencies/cub/cub "$CONDA_PREFIX/include/"

# Set include paths
export CPATH="$CONDA_PREFIX/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CUDA_HOME/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### Step 4: Create Environment and Install Dependencies
```bash
# Create conda environment
conda create -n blade_env python=3.9.19
conda activate blade_env

# Install PyTorch and dependencies
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath numpy==1.24.4 wandb
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# Install additional dependencies
pip install matplotlib==3.8.4 colorama requests huggingface-hub safetensors pillow six click openxlab
pip install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 packaging omegaconf ipdb ftfy regex
pip install json_tricks terminaltables modelindex prettytable albumentations smplx==0.1.28
pip install debugpy numba yacs scikit-learn filterpy h5py trimesh scikit-image tensorboardx pyrender
pip install torchgeometry joblib boto3 easydict pycocotools colormap pytorch-transformers pickle5 plyfile
pip install timm pyglet future tensorboard cdflib ftfy einops tqdm numpy==1.23.1 mediapipe

# Install project dependencies
cd mmcv && MMCV_WITH_OPS=1 pip install -e . -v && pip install -e . && cd ..
cd sapiens/engine && pip install -e . -v && cd ../pretrain && pip install -e . -v && cd ../pose && pip install -e . -v && cd ../det && pip install -e . -v && cd ../seg && pip install -e . -v && cd ../.. && pip install -e .
pip install ffmpeg astropy easydev pandas rtree vedo codecov flake8 interrogate isort pytest surrogate xdoctest setuptools loguru open3d omegaconf

# Install custom ops
cd aios_repo/models/aios/ops && python setup.py build install && cd ../../../..
cd torch-trust-ncg && python setup.py install && cd ..
pip install numpy==1.23.1
```

### Step 5: Setup Environment Scripts
```bash
# Create activation scripts for NVRTC and toolchain
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

# NVRTC 11.8 script
cat > "$CONDA_PREFIX/etc/conda/activate.d/10_nvrtc11.sh" <<'EOS'
# Prefer NVRTC 11.8 shipped in this env (pip: nvidia-cuda-nvrtc-cu11==11.8.89)
NVRTC_LIB_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib"

# remember old value to restore on deactivate
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Prepend the env's NVRTC dir so libnvrtc.so.11.8 and libnvrtc-builtins.so.11.8 resolve here
export LD_LIBRARY_PATH="$NVRTC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOS

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/10_nvrtc11.sh" <<'EOS'
# restore LD_LIBRARY_PATH
if [ -n "${_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
  unset _OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOS

# GCC-11 toolchain script
cat > "$CONDA_PREFIX/etc/conda/activate.d/20_cuda11_toolchain.sh" <<'EOS'
# Use conda-forge gcc-11 if installed
if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export NVCCFLAGS="--compiler-bindir=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi
EOS

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/20_cuda11_toolchain.sh" <<'EOS'
unset CC CXX CUDAHOSTCXX NVCCFLAGS
EOS
```

### Step 6: Fix EGL Issues
```bash
# Set EGL environment variables
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
[ -f "$CONDA_PREFIX/lib/libEGL.so" ] || ln -s "$CONDA_PREFIX/lib/libEGL.so.1" "$CONDA_PREFIX/lib/libEGL.so"
```

### Verification
```bash
# Test installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch3d; print('PyTorch3D installed successfully')"
```

> **Note:** Numpy version warnings**: These can be safely ignored as they don't affect functionality.
