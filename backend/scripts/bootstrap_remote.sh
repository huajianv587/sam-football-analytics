#!/bin/bash
set -euo pipefail

RUNTIME_DIR="${1:-${SAM2_RUNTIME_DIR:-$PWD/runtime}}"
ENV_DIR="$RUNTIME_DIR/env"

module load anaconda/25.5.1 cuda/12.8.0
mkdir -p "$RUNTIME_DIR/checkpoints" "$RUNTIME_DIR/torch-cache" "$RUNTIME_DIR/conda-pkgs" "$RUNTIME_DIR/pip-cache" "$RUNTIME_DIR/easyocr" "$RUNTIME_DIR/triton-cache" "$RUNTIME_DIR/torchinductor-cache"
export CONDA_PKGS_DIRS="$RUNTIME_DIR/conda-pkgs"
export PIP_CACHE_DIR="$RUNTIME_DIR/pip-cache"
export TORCH_HOME="$RUNTIME_DIR/torch-cache"
export EASYOCR_MODULE_PATH="$RUNTIME_DIR/easyocr"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" python=3.11
fi

conda install -y -p "$ENV_DIR" -c conda-forge ffmpeg gxx_linux-64=13
conda run -p "$ENV_DIR" pip install --upgrade pip
conda run -p "$ENV_DIR" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV_DIR" pip install easyocr opencv-python-headless scipy scikit-learn

if [[ ! -s "$RUNTIME_DIR/easyocr/model/english_g2.pth" ]]; then
  conda run -p "$ENV_DIR" python -c 'import easyocr; easyocr.Reader(["en"], gpu=False)'
fi

if [[ ! -d "$RUNTIME_DIR/sam2/.git" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "$RUNTIME_DIR/sam2"
else
  git -C "$RUNTIME_DIR/sam2" pull --ff-only
fi
export CC="$ENV_DIR/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_DIR/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="8.6"
SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 conda run -p "$ENV_DIR" pip install --no-build-isolation -e "$RUNTIME_DIR/sam2"

if [[ ! -s "$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt" ]]; then
  curl -fL \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -o "$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt"
fi

echo "SAM 2 runtime ready at $RUNTIME_DIR"
