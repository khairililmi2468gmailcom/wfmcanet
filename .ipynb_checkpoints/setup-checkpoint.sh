#!/bin/bash
# ============================================================
# setup.sh — jalankan SEKALI sebelum training pertama
# ============================================================
# Usage:
#   source /mnt/gpu17/miniconda3/bin/activate
#   conda activate rw-env
#   cd /mnt/gpu17/segilmi
#   bash setup.sh
# ============================================================

set -e   # stop on error

echo "=============================================="
echo "  WF-MCANet Setup — H100 Server"
echo "=============================================="

# 1. Set Kaggle token
export KAGGLE_API_TOKEN="KGAT_ea6881d628c7ca0551fcce57100d121c"

# Tulis kaggle.json agar CLI bisa baca
mkdir -p ~/.kaggle
# Token format KGAT_xxx: username perlu diset terpisah
# Ganti "your_kaggle_username" dengan username Kaggle Anda
KAGGLE_USER="${KAGGLE_USERNAME:-kibrobro}"
echo "{\"username\":\"${KAGGLE_USER}\",\"key\":\"${KAGGLE_API_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "kaggle.json created at ~/.kaggle/kaggle.json"

# 2. Install Python packages
echo ""
echo "Installing packages..."
pip install -q --upgrade timm
pip install -q medpy
pip install -q kaggle
pip install -q "albumentations==1.3.1"

echo ""
echo "Package versions:"
python -c "
import timm, medpy, albumentations, torch
print(f'  torch          {torch.__version__}')
print(f'  timm           {timm.__version__}')
print(f'  albumentations {albumentations.__version__}')
print(f'  mscan_t in timm: {\"mscan_t\" in timm.list_models()}')
print(f'  mit_b1  in timm: {\"mit_b1\" in timm.list_models()}')
"

# 3. Buat folder struktur
echo ""
echo "Creating directories..."
mkdir -p /mnt/gpu17/segilmi/{data/ISIC2018,data/Kvasir-SEG,results,checkpoints}
echo "  /mnt/gpu17/segilmi/data/ISIC2018"
echo "  /mnt/gpu17/segilmi/data/Kvasir-SEG"
echo "  /mnt/gpu17/segilmi/results"
echo "  /mnt/gpu17/segilmi/checkpoints"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "  Sekarang jalankan training dengan:"
echo ""
echo "    tmux new -s wfmcnet"
echo "    source /mnt/gpu17/miniconda3/bin/activate"
echo "    conda activate rw-env"
echo "    cd /mnt/gpu17/segilmi"
echo "    export KAGGLE_API_TOKEN=KGAT_ea6881d628c7ca0551fcce57100d121c"
echo "    python train.py 2>&1 | tee -a train.log"
echo "    # Ctrl+B  D   untuk detach"
echo "=============================================="
