#!/usr/bin/env bash
# AutoDL Environment Setup for CARE Major Revision Benchmark Suite (RTX 4090 24GB)
# Run in NO-GPU mode first (¥0.1/hr) to save cost!
set -euo pipefail

DATA_DISK="/root/autodl-tmp/care_eval"
mkdir -p "${DATA_DISK}/weights" "${DATA_DISK}/repos" "${DATA_DISK}/datasets" "${DATA_DISK}/runs"

# Enable AutoDL academic download proxy if available
if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

echo "[1/4] Installing Python dependencies (official PyPI package: vot-toolkit)..."
pip install --no-cache-dir hydra-core iopath timm loguru opencv-python-headless pillow scipy pyyaml tqdm transformers safetensors vot-toolkit

echo "[2/4] Downloading SAM 2.1 Tiny & Base+ official checkpoints..."
cd "${DATA_DISK}/weights"
if [ ! -f sam2.1_hiera_tiny.pt ]; then
    wget -c https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
fi
if [ ! -f sam2.1_hiera_base_plus.pt ]; then
    wget -c https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
fi
sha256sum sam2.1_hiera_tiny.pt sam2.1_hiera_base_plus.pt | tee checksums.sha256

echo "[3/4] Cloning separate official host source trees (Meta SAM 2, SAMURAI, DAM4SAM)..."
cd "${DATA_DISK}/repos"
if [ ! -d sam2_meta ]; then
    git clone https://github.com/facebookresearch/sam2.git sam2_meta
fi
if [ ! -d samurai ]; then
    git clone https://github.com/yangchris11/samurai.git samurai
fi
if [ ! -d DAM4SAM ]; then
    git clone https://github.com/jovanavidenovic/DAM4SAM.git DAM4SAM
fi

# Freeze exact git commit hashes of every host tree for reproducibility
{
    echo "sam2_meta: $(git -C "${DATA_DISK}/repos/sam2_meta" rev-parse HEAD)"
    echo "samurai:   $(git -C "${DATA_DISK}/repos/samurai" rev-parse HEAD)"
    echo "DAM4SAM:   $(git -C "${DATA_DISK}/repos/DAM4SAM" rev-parse HEAD)"
} | tee "${DATA_DISK}/repos/repo_commits.txt"

echo "[4/4] Verifying PyTorch CUDA, imports & official vot-toolkit version..."
python -c "import torch, hydra, iopath, loguru, cv2, vot; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'vot-toolkit:', vot.__version__)"
echo "SETUP_OK: ${DATA_DISK}"
