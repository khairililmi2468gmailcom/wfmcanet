#!/usr/bin/env bash
# =============================================================================
# run_demo.sh — WF-MCANet Demo Launcher
#
# CARA PAKAI:
#
#   1. Checkpoint D4g (terbaru, MSCAN-T backbone, sesuai paper):
#      bash run_demo.sh -c checkpoints/D4g_WFMCANet_best.pt
#
#   2. Checkpoint D4_Optimized:
#      bash run_demo.sh -c checkpoints/D4_Optimized_best.pt
#
#   3. Tanpa checkpoint (random init — demo UI saja):
#      bash run_demo.sh
#
#   4. Port custom:
#      bash run_demo.sh -c checkpoints/D4g_WFMCANet_best.pt -p 8080
#
# Setelah running, buka: http://localhost:5000
# Atau buka index.html langsung di browser (double-click).
#
# Catatan: MSCAN-T pretrained weights (~60MB) akan dicari di:
#   ~/.cache/mscan_t_imagenet.pth
# Jika ada, otomatis diload. Jika tidak ada, gunakan random init.
# =============================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

CKPT=""; PORT=5000
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--checkpoint) CKPT="$2"; shift 2 ;;
    -p|--port)       PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║    WF-MCANet Demo  ·  Segmentasi Medis   ║"
echo "  ║    MSCAN-T · MultiScale WFM · MCABlock   ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Cek dependensi
python3 -c "import flask,flask_cors,PIL,torch,torchvision" 2>/dev/null || {
  echo -e "${YELLOW}[install]${NC} pip install flask flask-cors pillow torch torchvision..."
  pip install flask flask-cors pillow torch torchvision --break-system-packages -q
}

if [ -n "$CKPT" ]; then
  [ -f "$CKPT" ] || { echo -e "${RED}[error]${NC} Checkpoint tidak ada: $CKPT"; exit 1; }
  echo -e "${GREEN}[model]${NC} Checkpoint: $CKPT"
  CMD="python3 app.py --checkpoint \"$CKPT\" --port $PORT"
else
  echo -e "${YELLOW}[model]${NC} Tidak ada checkpoint → random init"
  echo -e "         Untuk demo dengan bobot trained:"
  echo -e "         bash run_demo.sh -c checkpoints/D4g_WFMCANet_best.pt"
  CMD="python3 app.py --port $PORT"
fi

echo ""
echo -e "${GREEN}[url]${NC}  API  : http://localhost:$PORT"
echo -e "${GREEN}[url]${NC}  UI   : buka ${DIR}/index.html di browser"
echo ""
echo -e "  Ctrl+C untuk berhenti"
echo ""
eval "$CMD"
