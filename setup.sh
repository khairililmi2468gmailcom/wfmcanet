#!/usr/bin/env bash
# =============================================================================
#  WF-MCANet — dataset download + environment setup (run on the H100 server)
# =============================================================================
#  SECURITY: never hard-code your Kaggle token in this file or in logs.
#  Your previously pasted token KGAT_... is now exposed — REVOKE it at
#  https://www.kaggle.com/settings  ->  "Expire API Token", then create a new one.
#
#  Before running, export your credentials in the shell (they are NOT stored here):
#     export KAGGLE_USERNAME=your_kaggle_username
#     export KAGGLE_KEY=your_new_kaggle_key
#     ./setup.sh
# =============================================================================
set -e

BASE=/mnt/gpu17/segilmi
DATA=$BASE/data
ISIC=$DATA/ISIC2018
KV=$DATA/Kvasir-SEG
mkdir -p "$ISIC" "$KV" "$BASE/results" "$BASE/checkpoints"

echo "== [1/3] Installing Python dependencies =="
pip install -q timm albumentations medpy matplotlib pandas scipy pillow \
            opencv-python-headless kaggle PyWavelets

echo "== [2/3] Configuring Kaggle credentials (from environment) =="
mkdir -p ~/.kaggle
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
  printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "   kaggle.json written."
else
  echo "   WARNING: KAGGLE_USERNAME / KAGGLE_KEY not set. Export them and re-run." >&2
fi

echo "== [3/3] Downloading datasets =="
if [ ! -d "$ISIC/ISIC2018_Task1-2_Training_Input" ]; then
  echo "   Downloading ISIC 2018 ..."
  ( cd "$ISIC" && \
    kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation && \
    unzip -q isic2018-challenge-task1-data-segmentation.zip && \
    rm -f isic2018-challenge-task1-data-segmentation.zip )
else
  echo "   ISIC 2018 already present."
fi

if [ ! -d "$KV/images" ]; then
  echo "   Downloading Kvasir-SEG ..."
  ( cd "$KV" && \
    kaggle datasets download -d debeshjha1/kvasirseg && \
    unzip -q kvasirseg.zip && rm -f kvasirseg.zip )
  for n in kvasir-seg Kvasir-SEG kvasirseg; do
    if [ -d "$KV/$n" ]; then
      mv "$KV/$n/images" "$KV/" 2>/dev/null || true
      mv "$KV/$n/masks"  "$KV/" 2>/dev/null || true
    fi
  done
else
  echo "   Kvasir-SEG already present."
fi

echo ""
echo "== Setup complete. Next, generate the real figures + metrics: =="
echo "   python generate_figures.py --checkpoints $BASE/checkpoints --out $BASE/journal_assets"