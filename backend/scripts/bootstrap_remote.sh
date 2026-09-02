#!/bin/bash
set -euo pipefail

RUNTIME_DIR="${1:-${SAM2_RUNTIME_DIR:-$PWD/runtime}}"
ENV_DIR="$RUNTIME_DIR/env"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

module load anaconda/25.5.1 cuda/12.8.0
mkdir -p "$RUNTIME_DIR/checkpoints" "$RUNTIME_DIR/torch-cache" "$RUNTIME_DIR/conda-pkgs" "$RUNTIME_DIR/pip-cache" "$RUNTIME_DIR/easyocr" "$RUNTIME_DIR/triton-cache" "$RUNTIME_DIR/torchinductor-cache" "$RUNTIME_DIR/cache/matplotlib"
export CONDA_PKGS_DIRS="$RUNTIME_DIR/conda-pkgs"
export PIP_CACHE_DIR="$RUNTIME_DIR/pip-cache"
export TORCH_HOME="$RUNTIME_DIR/torch-cache"
export EASYOCR_MODULE_PATH="$RUNTIME_DIR/easyocr"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export MPLCONFIGDIR="$RUNTIME_DIR/cache/matplotlib"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" python=3.11
fi

conda install -y -p "$ENV_DIR" -c conda-forge ffmpeg gxx_linux-64=13
conda run -p "$ENV_DIR" pip install --upgrade pip
conda run -p "$ENV_DIR" pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV_DIR" pip install easyocr==1.7.2 opencv-python-headless==5.0.0.93 scipy==1.17.1 scikit-learn==1.9.0

if [[ ! -s "$RUNTIME_DIR/easyocr/model/english_g2.pth" ]]; then
  conda run -p "$ENV_DIR" python -c 'import easyocr; easyocr.Reader(["en"], gpu=False)'
fi

if [[ ! -d "$RUNTIME_DIR/sam2/.git" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "$RUNTIME_DIR/sam2"
fi
git -C "$RUNTIME_DIR/sam2" fetch origin "$SAM2_COMMIT"
git -C "$RUNTIME_DIR/sam2" checkout --detach "$SAM2_COMMIT"
export CC="$ENV_DIR/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_DIR/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="8.6"
SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 conda run -p "$ENV_DIR" pip install --no-build-isolation -e "$RUNTIME_DIR/sam2"

if [[ ! -s "$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt" ]]; then
  curl -fL --retry 12 --retry-delay 5 --retry-all-errors \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
    -o "$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt"
fi
echo "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  $RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt" | sha256sum -c -

if [[ ! -s "$RUNTIME_DIR/checkpoints/sam2.1_hiera_base_plus.pt" ]]; then
  curl -fL --retry 12 --retry-delay 5 --retry-all-errors \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
    -o "$RUNTIME_DIR/checkpoints/sam2.1_hiera_base_plus.pt"
fi
echo "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5  $RUNTIME_DIR/checkpoints/sam2.1_hiera_base_plus.pt" | sha256sum -c -

echo "SAM 2 runtime ready at $RUNTIME_DIR"

ROOT_DIR="$(dirname "$RUNTIME_DIR")"
GSR_RUNTIME_DIR="$ROOT_DIR/gsr-runtime"
GSR_ENV_DIR="$GSR_RUNTIME_DIR/env"
SOCCERMASTER_COMMIT="2e5619712d93f634b841aaf37231cd9fceb6b262"
mkdir -p "$GSR_RUNTIME_DIR" "$GSR_RUNTIME_DIR/pretrained_models/yolo" \
  "$GSR_RUNTIME_DIR/pretrained_models/reid" "$GSR_RUNTIME_DIR/pretrained_models/calibration" \
  "$GSR_RUNTIME_DIR/pretrained_models/jn" \
  "$GSR_RUNTIME_DIR/cache/matplotlib"
export XDG_CACHE_HOME="$GSR_RUNTIME_DIR/cache"
export MPLCONFIGDIR="$GSR_RUNTIME_DIR/cache/matplotlib"
export YOLO_CONFIG_DIR="$GSR_RUNTIME_DIR/cache/ultralytics"
export HF_HOME="$GSR_RUNTIME_DIR/cache/huggingface"
mkdir -p "$YOLO_CONFIG_DIR" "$HF_HOME"

if [[ ! -d "$GSR_RUNTIME_DIR/SoccerMaster/.git" ]]; then
  git clone https://github.com/haolinyang-hlyang/SoccerMaster.git "$GSR_RUNTIME_DIR/SoccerMaster"
fi
git -C "$GSR_RUNTIME_DIR/SoccerMaster" fetch origin "$SOCCERMASTER_COMMIT"
git -C "$GSR_RUNTIME_DIR/SoccerMaster" checkout --detach "$SOCCERMASTER_COMMIT"

if [[ ! -x "$GSR_ENV_DIR/bin/python" ]]; then
  conda create -y -p "$GSR_ENV_DIR" python=3.10.16
fi
conda run -p "$GSR_ENV_DIR" pip install --upgrade pip setuptools==78.1.1
conda run -p "$GSR_ENV_DIR" pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -p "$GSR_ENV_DIR" pip install \
  -e "$GSR_RUNTIME_DIR/SoccerMaster/codes/tracklab" \
  albumentations==1.4.19 huggingface_hub
conda run -p "$GSR_ENV_DIR" pip install --no-deps \
  -e "$GSR_RUNTIME_DIR/SoccerMaster/codes/sn-gamestate"
conda run -p "$GSR_ENV_DIR" pip install \
  -e "$GSR_RUNTIME_DIR/SoccerMaster/codes/sn-gamestate/plugins/calibration"
conda run -p "$GSR_ENV_DIR" pip install \
  easyocr==1.7.2 soccernet==0.1.62 \
  "git+https://github.com/VlSomers/prtreid@30617a75967e84d5d516959c4b84cbeea6f56493"
INSTALL_ROLE_MODEL="${INSTALL_ROLE_MODEL:-false}"
if [[ "$INSTALL_ROLE_MODEL" == "true" ]]; then
  conda run -p "$GSR_ENV_DIR" pip install \
    transformers==4.51.3 accelerate==1.2.1 qwen-vl-utils==0.0.11
fi

# TrackLab 1.1.2 imports its optional standalone StrongSORT wrapper even when
# the configured tracker is BPBReIDStrongSORT. That optional wrapper ships
# without its ReID subpackage in this pinned release, so remove only the eager
# import; the BPBReIDStrongSORT wrapper and plugin remain unchanged.
sed -i -e '/^from \.strong_sort_api import StrongSORT$/d' \
  -e '/^from \.bot_sort_api import BotSORT$/d' \
  -e '/^from \.deep_oc_sort_api import DeepOCSORT$/d' \
  "$GSR_RUNTIME_DIR/SoccerMaster/codes/tracklab/tracklab/wrappers/track/__init__.py"

# PnLCalib uses label-based Series indexing that stopped meaning "first row"
# with current pandas. Its calibration batch is explicitly size one, so make
# that intent positional and reproducible without vendoring third-party code.
sed -i \
  -e 's/metadatas\["keypoints"\]\[0\]/metadatas["keypoints"].iloc[0]/' \
  -e 's/metadatas\["lines_det"\]\[0\]/metadatas["lines_det"].iloc[0]/' \
  "$GSR_RUNTIME_DIR/SoccerMaster/codes/sn-gamestate/sn_gamestate/calibration/pnlcalib.py"

# TrackLab decodes dataset frames as RGB while this Ultralytics entrypoint
# interprets ndarray inputs as BGR. Convert only at the detector boundary and
# pass the configured confidence into prediction; otherwise dark-uniform
# officials can disappear even when the football weight detects them.
sed -i \
  's/results_by_image = self\.model(images)$/results_by_image = self.model([image[..., ::-1].copy() for image in images], conf=float(self.cfg.min_confidence))/' \
  "$GSR_RUNTIME_DIR/SoccerMaster/codes/sn-gamestate/sn_gamestate/detect_multiple/yolov8_person_api.py"

if [[ ! -s "$GSR_RUNTIME_DIR/pretrained_models/yolo/yolo_v8x6_finetuned.pt" ]]; then
  conda run -p "$GSR_ENV_DIR" hf download xleprime/SoccerMaster \
    yolo_v8x6_finetuned.pt \
    --revision 0d94573662dbd678df22aa5c61ae77f474b939c9 \
    --local-dir "$GSR_RUNTIME_DIR/pretrained_models/yolo"
fi
echo "c85259af82ae919294e9ec87457d73646682159a5511ec1df033fed84b15d03d  $GSR_RUNTIME_DIR/pretrained_models/yolo/yolo_v8x6_finetuned.pt" | sha256sum -c -

QWEN_REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN_DIR="$GSR_RUNTIME_DIR/pretrained_models/jn/Qwen2.5-VL-7B-Instruct"
if [[ "$INSTALL_ROLE_MODEL" == "true" ]]; then
  if [[ ! -s "$QWEN_DIR/config.json" ]]; then
    conda run -p "$GSR_ENV_DIR" hf download Qwen/Qwen2.5-VL-7B-Instruct \
      --revision "$QWEN_REVISION" --local-dir "$QWEN_DIR"
  fi
  echo "$QWEN_REVISION" > "$QWEN_DIR/REVISION"
  find "$QWEN_DIR" -type f ! -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum \
    > "$QWEN_DIR/SHA256SUMS"
fi

"$GSR_ENV_DIR/bin/python" "$SCRIPT_DIR/download_model.py" \
  "https://zenodo.org/api/records/10653453/files/prtreid-soccernet-baseline.pth.tar/content" \
  "$GSR_RUNTIME_DIR/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar" \
  9633825232bc89f23a94522c5561650e
"$GSR_ENV_DIR/bin/python" "$SCRIPT_DIR/download_model.py" \
  "https://zenodo.org/api/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth/content" \
  "$GSR_RUNTIME_DIR/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth" \
  58ea12b0420aa3adaa2f74114c9f9721
"$GSR_ENV_DIR/bin/python" "$SCRIPT_DIR/download_model.py" \
  "https://zenodo.org/api/records/14046275/files/pnl_SV_kp/content" \
  "$GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_kp" \
  322d4a6c82d2966ea88b69963ba85f07
"$GSR_ENV_DIR/bin/python" "$SCRIPT_DIR/download_model.py" \
  "https://zenodo.org/api/records/14046275/files/pnl_SV_lines/content" \
  "$GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_lines" \
  270b94527c9e817bc32edd54c8e47b62

echo "9633825232bc89f23a94522c5561650e  $GSR_RUNTIME_DIR/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar" | md5sum -c -
echo "58ea12b0420aa3adaa2f74114c9f9721  $GSR_RUNTIME_DIR/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth" | md5sum -c -
echo "322d4a6c82d2966ea88b69963ba85f07  $GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_kp" | md5sum -c -
echo "270b94527c9e817bc32edd54c8e47b62  $GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_lines" | md5sum -c -

sha256sum "$GSR_RUNTIME_DIR/pretrained_models/yolo/yolo_v8x6_finetuned.pt" \
  "$GSR_RUNTIME_DIR/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar" \
  "$GSR_RUNTIME_DIR/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth" \
  "$GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_kp" \
  "$GSR_RUNTIME_DIR/pretrained_models/calibration/pnl_SV_lines" \
  "$RUNTIME_DIR/checkpoints/sam2.1_hiera_base_plus.pt" \
  "$RUNTIME_DIR/checkpoints/sam2.1_hiera_large.pt" \
  > "$ROOT_DIR/MODEL_SHA256SUMS"

echo "Soccer game-state runtime ready at $GSR_RUNTIME_DIR"
