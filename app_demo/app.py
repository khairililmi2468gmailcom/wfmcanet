"""
WF-MCANet Demo API — Config B: B_MCANet_WFM_Only
Backbone  : EfficientNet-B3 (timm, ~10.27M params)
Output    : 2-class softmax → foreground channel (class 1)
Checkpoint: B_MCANet_WFM_Only_best.pt
"""

import os, time, io, base64, argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

from model import WFMCANet, load_model

INPUT_SIZE = 512   # train.py uses IMG_SIZE=512
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

STATIC_DIR = Path(__file__).parent
app        = Flask(__name__, static_folder=None)
CORS(app)

MODEL     = None
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_NAME = 'random init'


# ── Static files ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(str(STATIC_DIR), 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert('RGB').resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).float()

def prob_to_png_b64(prob_hw, orig_size=None):
    """prob_hw: H×W float [0,1] → greyscale PNG base64"""
    m8  = (prob_hw * 255).clip(0,255).astype(np.uint8)
    img = Image.fromarray(m8, 'L')
    if orig_size:
        img = img.resize(orig_size, Image.BILINEAR)
    buf = io.BytesIO(); img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()

def make_overlay(pil_orig, mask_hw, alpha=0.42, color=(0,210,110)):
    """Semi-transparent mask overlay + red boundary on original image."""
    rgb   = np.array(pil_orig.convert('RGB').resize(
                (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR), dtype=np.float32)
    m     = mask_hw.astype(np.float32)
    for c, col in enumerate(color):
        rgb[:,:,c] = rgb[:,:,c] * (1 - alpha*m) + col * alpha*m
    m8    = (mask_hw * 255).astype(np.uint8)
    erode = np.array(Image.fromarray(m8,'L').filter(ImageFilter.MinFilter(3)))
    bound = (m8.astype(int) - erode.astype(int)) > 30
    rgb[bound] = [255, 60, 60]
    buf = io.BytesIO()
    Image.fromarray(rgb.clip(0,255).astype(np.uint8)).save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── API Routes ────────────────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    p   = sum(p.numel() for p in MODEL.parameters()) / 1e6
    bn  = getattr(MODEL.encoder, 'backbone_name', 'unknown')
    return jsonify({
        'status':     'ready',
        'model':      'WF-MCANet Config B (MCANet + WFM)',
        'checkpoint': CKPT_NAME,
        'device':     DEVICE,
        'gpu':        gpu,
        'params_M':   round(p, 2),
        'input_size': INPUT_SIZE,
        'backbone':   bn,
        'output':     '2-class softmax (foreground = class 1)',
        'paper_results': {
            'dice_isic2018': 0.9123,
            'hd95_isic2018': 23.52,
            'dice_zeroshot': 0.296,
            'hd95_zeroshot': 198.74,
        }
    })


@app.route('/api/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file      = request.files['image']
    threshold = float(request.form.get('threshold', 0.5))

    try:
        pil      = Image.open(file.stream)
        orig_w, orig_h = pil.size
        tensor   = preprocess(pil).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = MODEL(tensor)            # (1,2,H,W)
        ms = (time.perf_counter() - t0) * 1000

        # Foreground probability = softmax channel 1
        prob   = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # (H,W) [0,1]
        binary = (prob >= threshold).astype(np.float32)

        fg_px  = int(binary.sum())
        cov    = round(fg_px / binary.size * 100, 2)

        return jsonify({
            'overlay_b64':  make_overlay(pil, binary),
            'mask_b64':     prob_to_png_b64(prob, (orig_w, orig_h)),
            'inference_ms': round(ms, 1),
            'threshold':    threshold,
            'coverage_pct': cov,
            'fg_pixels':    fg_px,
            'total_pixels': binary.size,
            'orig_size':    [orig_w, orig_h],
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/load_checkpoint', methods=['POST'])
def load_checkpoint_route():
    global MODEL, CKPT_NAME
    if 'checkpoint' not in request.files:
        return jsonify({'error': 'No checkpoint file'}), 400
    f    = request.files['checkpoint']
    path = f'/tmp/{f.filename}'
    f.save(path)
    try:
        MODEL     = load_model(path, DEVICE)
        CKPT_NAME = f.filename
        p = sum(p.numel() for p in MODEL.parameters()) / 1e6
        return jsonify({'status': 'ok', 'loaded': f.filename, 'params_M': round(p,2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None)
    parser.add_argument('--port', '-p', type=int, default=5000)
    args = parser.parse_args()

    CKPT_NAME = Path(args.checkpoint).name if args.checkpoint else 'random init'
    print(f'[WF-MCANet] Device: {DEVICE}')
    MODEL = load_model(args.checkpoint, DEVICE)
    print(f'[WF-MCANet] Ready — http://localhost:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
