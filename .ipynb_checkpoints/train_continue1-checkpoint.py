#!/usr/bin/env python3
"""
WF-MCANet: Wavelet Frequency-Enhanced Cross-Axis Attention
for Boundary-Precise Medical Image Segmentation

Standalone training script — run inside tmux on H100 server.
All output is written to both terminal and LOG_FILE.

Usage:
    python train.py

Output:
    /mnt/gpu17/segilmi/results/   — metrics, CSV, plots
    /mnt/gpu17/segilmi/checkpoints/ — best model per config
    /mnt/gpu17/segilmi/train.log  — full training log
"""

# ============================================================
# 0. Imports & reproducibility
# ============================================================
import os, sys, json, time, math, random, shutil, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no display needed on server
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from medpy.metric.binary import hd95 as compute_hd95

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Paths  (semua di /mnt/gpu17/segilmi/)
# ============================================================
BASE_DIR       = Path('/mnt/gpu17/segilmi')
DATA_DIR       = BASE_DIR / 'data'
ISIC_DIR       = DATA_DIR / 'ISIC2018'
KVASIR_DIR     = DATA_DIR / 'Kvasir-SEG'
RESULTS_DIR    = BASE_DIR / 'results'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
LOG_FILE       = BASE_DIR / 'train.log'

for d in [DATA_DIR, ISIC_DIR, KVASIR_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Logger — prints to terminal AND writes to train.log
# ============================================================
class Tee:
    """Mirror stdout to a file."""
    def __init__(self, path):
        self.file = open(path, 'a', buffering=1)   # line-buffered
        self.terminal = sys.stdout
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee(LOG_FILE)

def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ============================================================
# 3. Environment info + auto-fix timm for mscan_t
# ============================================================
import subprocess as _sp

log('=' * 62)
log('  WF-MCANet Training — H100 Server')
log(f'  Started : {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'  Device  : {DEVICE}')
log(f'  PyTorch : {torch.__version__}')
if torch.cuda.is_available():
    log(f'  GPU     : {torch.cuda.get_device_name(0)}')
    log(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
log(f'  Log     : {LOG_FILE}')

# Check /dev/shm size (shared memory — important for Docker)
try:
    shm = _sp.run(['df', '-h', '/dev/shm'], capture_output=True, text=True)
    shm_line = [l for l in shm.stdout.splitlines() if '/dev/shm' in l]
    log(f'  /dev/shm: {shm_line[0].split()[1] if shm_line else "unknown"}')
except Exception:
    pass

# Auto-upgrade timm if mscan_t not available
if 'mscan_t' not in timm.list_models():
    log('  mscan_t not in timm — upgrading timm ...')
    _sp.run([sys.executable, '-m', 'pip', 'install', '-q',
             '--upgrade', 'timm'], check=True)
    # Reload timm after upgrade
    import importlib
    import timm as _timm_new
    importlib.reload(_timm_new)
    # Replace the timm module reference in globals
    import sys as _sys
    _sys.modules['timm'] = _timm_new
    timm = _timm_new
    log(f'  timm upgraded to {timm.__version__}')
    log(f'  mscan_t now available: {"mscan_t" in timm.list_models()}')
else:
    log(f'  timm {timm.__version__} — mscan_t: OK')

log('=' * 62)

# ============================================================
# 4. Dataset download helpers
# ============================================================
import subprocess

def download_isic():
    images_dir = ISIC_DIR / 'ISIC2018_Task1-2_Training_Input'
    masks_dir  = ISIC_DIR / 'ISIC2018_Task1_Training_GroundTruth'
    if images_dir.exists() and len(list(images_dir.glob('*.jpg'))) >= 100:
        log(f'ISIC 2018 already present: {len(list(images_dir.glob("*.jpg")))} images')
        return images_dir, masks_dir

    log('Downloading ISIC 2018 via Kaggle API ...')
    # Kaggle token set via env var KAGGLE_API_TOKEN
    token = os.environ.get('KAGGLE_API_TOKEN', '')
    if token:
        # Write kaggle.json from token
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / 'kaggle.json'
        # token format: "username:key"  OR just the key (KGAT_...)
        if ':' in token:
            user, key = token.split(':', 1)
        else:
            # Try to extract from KGAT token (username embedded)
            user = os.environ.get('KAGGLE_USERNAME', 'user')
            key  = token
        kaggle_json.write_text(json.dumps({'username': user, 'key': key}))
        kaggle_json.chmod(0o600)

    prev = Path.cwd()
    os.chdir(ISIC_DIR)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'tschandl/isic2018-challenge-task1-data-segmentation'
    ], check=True)
    subprocess.run(['unzip', '-q',
        'isic2018-challenge-task1-data-segmentation.zip'], check=True)
    (ISIC_DIR / 'isic2018-challenge-task1-data-segmentation.zip').unlink(missing_ok=True)
    os.chdir(prev)
    log(f'ISIC 2018 downloaded: {len(list(images_dir.glob("*.jpg")))} images')
    return images_dir, masks_dir


def download_kvasir():
    kv_img = KVASIR_DIR / 'images'
    kv_msk = KVASIR_DIR / 'masks'

    if kv_img.exists() and len(list(kv_img.glob('*.jpg'))) >= 50:
        log(f'Kvasir-SEG already present: {len(list(kv_img.glob("*.jpg")))} images')
        return kv_img, kv_msk

    log('Downloading Kvasir-SEG via Kaggle API (debeshjha1/kvasirseg) ...')
    prev = Path.cwd()
    os.chdir(KVASIR_DIR)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'debeshjha1/kvasirseg'
    ], check=True)

    # Find the downloaded zip (name may vary)
    zips = list(KVASIR_DIR.glob('*.zip'))
    if not zips:
        raise RuntimeError('Kaggle download finished but no zip found in Kvasir dir.')
    zip_path = zips[0]
    log(f'Extracting {zip_path.name} ...')
    subprocess.run(['unzip', '-q', str(zip_path)], check=True)
    zip_path.unlink(missing_ok=True)

    # Handle possible nested folder structures
    for nested_name in ['kvasir-seg', 'Kvasir-SEG', 'kvasirseg']:
        nested = KVASIR_DIR / nested_name
        if nested.exists() and (nested / 'images').exists():
            shutil.move(str(nested / 'images'), str(KVASIR_DIR))
            shutil.move(str(nested / 'masks'),  str(KVASIR_DIR))
            shutil.rmtree(str(nested))
            break

    os.chdir(prev)
    n = len(list(kv_img.glob('*.jpg')))
    log(f'Kvasir-SEG ready: {n} images')
    return kv_img, kv_msk

# ============================================================
# 5. Download datasets
# ============================================================
isic_images, isic_masks = download_isic()
kvasir_images, kvasir_masks = download_kvasir()

# ============================================================
# 6. Train/test split (2074 / 520, same as MCANet paper)
# ============================================================
split_file = DATA_DIR / 'isic2018_split.json'
if split_file.exists():
    split = json.loads(split_file.read_text())
    train_pairs = split['train']
    test_pairs  = split['test']
    log(f'Loaded split: train={len(train_pairs)}, test={len(test_pairs)}')
else:
    all_imgs = sorted(isic_images.glob('*.jpg'))
    all_msks = [isic_masks / (img.stem + '_segmentation.png') for img in all_imgs]
    pairs = [(str(i), str(m)) for i, m in zip(all_imgs, all_msks) if m.exists()]
    random.shuffle(pairs)
    train_pairs = pairs[:2074]
    test_pairs  = pairs[2074:]
    split_file.write_text(json.dumps({'train': train_pairs, 'test': test_pairs}))
    log(f'Created split: train={len(train_pairs)}, test={len(test_pairs)}')

kvasir_pairs = [
    (str(img), str(kvasir_masks / img.name))
    for img in sorted(kvasir_images.glob('*.jpg'))
    if (kvasir_masks / img.name).exists()
]
log(f'Kvasir-SEG pairs: {len(kvasir_pairs)}')

# ============================================================
# 7. Dataset & DataLoaders
# ============================================================
IMG_SIZE   = 512
BATCH_SIZE = 8

# Docker containers have limited /dev/shm (shared memory).
# num_workers > 0 uses shared memory for inter-process data transfer
# and causes Bus error when /dev/shm is too small.
# Fix: num_workers=0 (load in main process) + pin_memory=False.
# On H100 with NVMe storage this is still fast enough.
NUM_WORKERS = 0
PIN_MEMORY  = False

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class SegDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask  = np.array(Image.open(msk_path).convert('L'))
        mask  = (mask > 127).astype(np.uint8)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask'].long()
        return image, mask


train_loader  = DataLoader(SegDataset(train_pairs, train_transform),
                           batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader   = DataLoader(SegDataset(test_pairs,  val_transform),
                           batch_size=4, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
kvasir_loader = DataLoader(SegDataset(kvasir_pairs, val_transform),
                           batch_size=4, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

log(f'Train batches : {len(train_loader)} ({len(train_loader.dataset)} samples)')
log(f'Test batches  : {len(test_loader)} ({len(test_loader.dataset)} samples)')
log(f'Kvasir batches: {len(kvasir_loader)} ({len(kvasir_loader.dataset)} samples)')

# ============================================================
# 8. Encoder — dynamic backbone selection
# ============================================================
_CANDIDATES = [
    ('mscan_t',         True,  (0,1,2,3), 'MSCAN-T (original MCANet backbone)'),
    ('mscan_s',         True,  (0,1,2,3), 'MSCAN-S'),
    ('mit_b1',          True,  (0,1,2,3), 'MiT-B1 Mix Transformer'),
    ('mit_b0',          True,  (0,1,2,3), 'MiT-B0 Mix Transformer (tiny)'),
    ('efficientnet_b3', True,  (1,2,3,4), 'EfficientNet-B3 (fallback)'),
    ('resnet50',        True,  (1,2,3,4), 'ResNet-50 (universal fallback)'),
]

_available     = set(timm.list_models())
_chosen_name   = None
_chosen_indices = None
log(f'timm {timm.__version__} — {len(_available)} models registered')

for _name, _fo, _idx, _note in _CANDIDATES:
    if _name not in _available:
        log(f'  skip {_name} (not in registry)')
        continue
    try:
        _m = timm.create_model(_name, pretrained=False, features_only=_fo, out_indices=_idx)
        _chosen_name    = _name
        _chosen_indices = _idx
        del _m
        log(f'  SELECTED: {_name} — {_note}')
        break
    except Exception as e:
        log(f'  skip {_name}: {e}')

assert _chosen_name, 'No backbone found — install timm >= 0.9.12'


class MSCANEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            _chosen_name, pretrained=pretrained,
            features_only=True, out_indices=_chosen_indices)
        self.channels      = self.backbone.feature_info.channels()
        self.backbone_name = _chosen_name
        log(f'Encoder loaded: {_chosen_name}  channels={self.channels}')
    def forward(self, x):
        return self.backbone(x)

# ============================================================
# 9. MCA Decoder components
# ============================================================
class DWConv1D(nn.Module):
    def __init__(self, dim, kernel_size, axis='x'):
        super().__init__()
        if axis == 'x':
            self.conv = nn.Conv2d(dim, dim, (1, kernel_size),
                                  padding=(0, kernel_size//2), groups=dim)
        else:
            self.conv = nn.Conv2d(dim, dim, (kernel_size, 1),
                                  padding=(kernel_size//2, 0), groups=dim)
    def forward(self, x): return self.conv(x)


class MultiScaleAxisConv(nn.Module):
    def __init__(self, dim, axis='x'):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.conv7  = DWConv1D(dim, 7,  axis=axis)
        self.conv11 = DWConv1D(dim, 11, axis=axis)
        self.conv21 = DWConv1D(dim, 21, axis=axis)
        self.proj   = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        x_n = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.proj(self.conv7(x_n) + self.conv11(x_n) + self.conv21(x_n))


class CrossAxisAttention(nn.Module):
    def __init__(self, dim, num_heads=8, axis='y'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.axis      = axis
        self.to_q  = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.out   = nn.Linear(dim, dim, bias=False)
    def forward(self, query_src, kv_src):
        B, C, H, W = query_src.shape
        nh, hd = self.num_heads, self.head_dim
        if self.axis == 'y':
            q  = query_src.permute(0,3,2,1).reshape(B*W, H, C)
            kv = kv_src.permute(0,3,2,1).reshape(B*W, H, C)
            seq = H
        else:
            q  = query_src.permute(0,2,3,1).reshape(B*H, W, C)
            kv = kv_src.permute(0,2,3,1).reshape(B*H, W, C)
            seq = W
        Q  = self.to_q(q).reshape(-1, seq, nh, hd).transpose(1,2)
        KV = self.to_kv(kv).reshape(-1, seq, 2, nh, hd).permute(2,0,3,1,4)
        K, V = KV[0], KV[1]
        attn = F.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        out  = (attn @ V).transpose(1,2).reshape(-1, seq, C)
        out  = self.out(out)
        if self.axis == 'y':
            out = out.reshape(B, W, H, C).permute(0,3,2,1)
        else:
            out = out.reshape(B, H, W, C).permute(0,3,1,2)
        return out


class MCABlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.x_conv = MultiScaleAxisConv(dim, axis='x')
        self.y_conv = MultiScaleAxisConv(dim, axis='y')
        self.top_ca = CrossAxisAttention(dim, num_heads, axis='y')
        self.bot_ca = CrossAxisAttention(dim, num_heads, axis='x')
        self.proj_t = nn.Conv2d(dim, dim, 1)
        self.proj_b = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        Fx = self.x_conv(x)
        Fy = self.y_conv(x)
        return self.proj_t(self.top_ca(Fy, Fx)) + self.proj_b(self.bot_ca(Fx, Fy)) + x

# ============================================================
# 10. Wavelet Frequency Module (WFM) — core novelty
# ============================================================
class WaveletFrequencyModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        haar = torch.tensor([
            [ 1, 1, 1, 1], [ 1, 1,-1,-1],
            [ 1,-1, 1,-1], [ 1,-1,-1, 1],
        ], dtype=torch.float32).view(4, 1, 2, 2) * 0.5
        self.register_buffer('haar', haar)
        self.channels = channels
        inner = max(channels * 4 // reduction, 8)
        self.ca_fc1 = nn.Linear(channels * 4, inner)
        self.ca_fc2 = nn.Linear(inner, channels * 4)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        sub  = F.conv2d(x.reshape(B*C, 1, H, W), self.haar,
                        stride=2).reshape(B, C*4, H//2, W//2)
        gap  = sub.mean(dim=[2,3])
        w    = torch.sigmoid(self.ca_fc2(F.relu(self.ca_fc1(gap))))
        sub  = sub * w.unsqueeze(-1).unsqueeze(-1)
        fhat = self.proj(sub)
        fhat = F.interpolate(fhat, size=(H, W), mode='bilinear', align_corners=False)
        return x + torch.sigmoid(self.gate) * fhat

# ============================================================
# 11. Full architectures
# ============================================================
def _make_proj(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1),
                         nn.BatchNorm2d(out_ch), nn.ReLU())


class MCANet(nn.Module):
    """MCANet baseline (faithful reproduction of Shao et al., MIR 2025)."""
    def __init__(self, num_classes=2, dec_channels=64, num_heads=8, pretrained=True):
        super().__init__()
        self.encoder = MSCANEncoder(pretrained=pretrained)
        ec = self.encoder.channels
        self.up_e2      = _make_proj(ec[1], dec_channels)
        self.up_e3      = _make_proj(ec[2], dec_channels)
        self.up_e4      = _make_proj(ec[3], dec_channels)
        self.reduce_conv= _make_proj(dec_channels * 3, dec_channels)
        self.mca        = MCABlock(dec_channels, num_heads)
        self.fuse_e1    = nn.Sequential(
            _make_proj(dec_channels + ec[0], dec_channels),
            nn.Conv2d(dec_channels, dec_channels, 1),
            nn.BatchNorm2d(dec_channels), nn.ReLU())
        self.head = nn.Conv2d(dec_channels, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        E1, E2, E3, E4 = self.encoder(x)
        ts = E1.shape[2:]
        e2 = F.interpolate(self.up_e2(E2), ts, mode='bilinear', align_corners=False)
        e3 = F.interpolate(self.up_e3(E3), ts, mode='bilinear', align_corners=False)
        e4 = F.interpolate(self.up_e4(E4), ts, mode='bilinear', align_corners=False)
        fused = self.reduce_conv(torch.cat([e2, e3, e4], 1))
        attn  = self.mca(fused)
        out   = self.fuse_e1(torch.cat([attn, E1], 1))
        out   = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        return self.head(out)


class WFMCANet(nn.Module):
    """WF-MCANet: MCANet + Wavelet Frequency Module + Edge Loss."""
    def __init__(self, num_classes=2, dec_channels=64, num_heads=8, pretrained=True):
        super().__init__()
        self.encoder = MSCANEncoder(pretrained=pretrained)
        ec = self.encoder.channels
        self.up_e2       = _make_proj(ec[1], dec_channels)
        self.up_e3       = _make_proj(ec[2], dec_channels)
        self.up_e4       = _make_proj(ec[3], dec_channels)
        self.reduce_conv = _make_proj(dec_channels * 3, dec_channels)
        self.wfm         = WaveletFrequencyModule(dec_channels, reduction=4)
        self.mca         = MCABlock(dec_channels, num_heads)
        self.fuse_e1     = nn.Sequential(
            _make_proj(dec_channels + ec[0], dec_channels),
            nn.Conv2d(dec_channels, dec_channels, 1),
            nn.BatchNorm2d(dec_channels), nn.ReLU())
        self.head = nn.Conv2d(dec_channels, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        E1, E2, E3, E4 = self.encoder(x)
        ts = E1.shape[2:]
        e2 = F.interpolate(self.up_e2(E2), ts, mode='bilinear', align_corners=False)
        e3 = F.interpolate(self.up_e3(E3), ts, mode='bilinear', align_corners=False)
        e4 = F.interpolate(self.up_e4(E4), ts, mode='bilinear', align_corners=False)
        fused    = self.reduce_conv(torch.cat([e2, e3, e4], 1))
        fused_wf = self.wfm(fused)
        attn     = self.mca(fused_wf)
        out      = self.fuse_e1(torch.cat([attn, E1], 1))
        out      = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        return self.head(out)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ============================================================
# 12. Loss functions
# ============================================================
def dice_loss(pred, target, smooth=1e-5):
    prob = torch.softmax(pred, 1)
    tgt  = torch.zeros_like(prob)
    tgt.scatter_(1, target.unsqueeze(1), 1)
    inter = (prob * tgt).sum(dim=(2,3))
    union = prob.sum(dim=(2,3)) + tgt.sum(dim=(2,3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def edge_loss(pred, target, weight=0.7):
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sy = sx.transpose(2,3)
    mf = target.float().unsqueeze(1)
    eg = (F.conv2d(mf, sx, padding=1).abs() +
          F.conv2d(mf, sy, padding=1).abs()).clamp(0,1)
    # Use lesion logit directly with BCEWithLogitsLoss — safe under autocast
    # (avoids unsafe binary_cross_entropy after softmax)
    pl_logit = pred[:, 1:2].float()   # cast to float32 for stability
    eg = eg.float()
    return weight * F.binary_cross_entropy_with_logits(pl_logit, eg)


def total_loss(pred, target, use_edge=True, edge_weight=0.7):
    loss = F.cross_entropy(pred, target) + dice_loss(pred, target)
    if use_edge:
        loss = loss + edge_loss(pred, target, edge_weight)
    return loss

# ============================================================
# 13. Metrics
# ============================================================
def compute_metrics(pred_logits, target_masks):
    pred_masks = torch.argmax(pred_logits, 1)
    dice_l, iou_l, hd95_l = [], [], []
    sm = 1e-5
    for p, t in zip(pred_masks, target_masks):
        p = p.cpu().numpy().astype(bool)
        t = t.cpu().numpy().astype(bool)
        inter = (p & t).sum()
        dice_l.append((2*inter + sm) / (p.sum() + t.sum() + sm))
        iou_l.append( (inter + sm) / ((p | t).sum() + sm) )
        if p.sum() > 0 and t.sum() > 0:
            try: hd95_l.append(compute_hd95(p, t))
            except: pass
    return {
        'dice': float(np.mean(dice_l)),
        'iou':  float(np.mean(iou_l)),
        'hd95': float(np.mean(hd95_l)) if hd95_l else float('nan'),
    }

# ============================================================
# 14. Training loop
# ============================================================
def train_one_epoch(model, loader, optimizer, use_edge, scaler):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = total_loss(logits, masks, use_edge=use_edge)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dice_l, iou_l, hd95_l = [], [], []
    t_total, n = 0.0, 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        t0 = time.time()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        torch.cuda.synchronize()
        t_total += time.time() - t0
        n += imgs.shape[0]
        m = compute_metrics(logits, masks)
        dice_l.append(m['dice']); iou_l.append(m['iou'])
        if not math.isnan(m['hd95']): hd95_l.append(m['hd95'])
    return {
        'dice': float(np.mean(dice_l)),
        'iou':  float(np.mean(iou_l)),
        'hd95': float(np.mean(hd95_l)) if hd95_l else float('nan'),
        'ms_per_image': 1000.0 * t_total / n,
    }


def run_experiment(config_name, model, use_edge_loss,
                   num_epochs=100, lr=2e-4, weight_decay=1e-4, eval_every=5):
    log(f'\n{"="*62}')
    log(f'  EXPERIMENT : {config_name}')
    log(f'  Edge loss  : {use_edge_loss}')
    tp, _ = count_params(model)
    log(f'  Parameters : {tp/1e6:.3f} M')
    log(f'  Epochs     : {num_epochs}  eval_every={eval_every}')
    log(f'{"="*62}')

    ckpt_path = CHECKPOINT_DIR / f'{config_name}_best.pt'
    log_path  = RESULTS_DIR    / f'{config_name}_log.json'

    opt     = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    scaler  = torch.cuda.amp.GradScaler()
    best    = 0.0
    history = defaultdict(list)
    t_start = time.time()

    for ep in range(1, num_epochs + 1):
        t0   = time.time()
        tloss = train_one_epoch(model, train_loader, opt, use_edge_loss, scaler)
        sched.step()
        elapsed = time.time() - t0

        do_eval = (ep % eval_every == 0) or ep == num_epochs or ep == 1
        if do_eval:
            m = evaluate(model, test_loader)
            history['epoch'].append(ep)
            history['train_loss'].append(tloss)
            history['dice'].append(m['dice'])
            history['iou'].append(m['iou'])
            history['hd95'].append(m['hd95'])
            star = ''
            if m['dice'] > best:
                best = m['dice']
                torch.save({'epoch': ep, 'model': model.state_dict(),
                            'metrics': m}, ckpt_path)
                star = '  <-- best'
            hd = f"{m['hd95']:.2f}" if not math.isnan(m['hd95']) else ' nan'
            log(f'Ep {ep:3d}/{num_epochs} | loss={tloss:.4f} | '
                f'Dice={m["dice"]:.4f} | IoU={m["iou"]:.4f} | '
                f'HD95={hd}px | {elapsed:.0f}s{star}')
        else:
            eta_m = int(elapsed * (num_epochs - ep) / 60)
            log(f'Ep {ep:3d}/{num_epochs} | loss={tloss:.4f} | '
                f'{elapsed:.0f}s/ep  ETA~{eta_m}min')

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    fm   = evaluate(model, test_loader)
    wall = (time.time() - t_start) / 60

    log(f'\n{"-"*62}')
    log(f'  FINAL [{config_name}]  (best @ epoch {ckpt["epoch"]})')
    log(f'  Dice      : {fm["dice"]:.4f}')
    log(f'  IoU       : {fm["iou"]:.4f}')
    log(f'  HD95      : {fm["hd95"]:.2f} px')
    log(f'  Inference : {fm["ms_per_image"]:.1f} ms/image')
    log(f'  Wall time : {wall:.1f} min')
    log(f'{"-"*62}')

    json.dump({'config': config_name, 'history': dict(history), 'final': fm},
              open(log_path, 'w'), indent=2)
    return fm, history

# ============================================================
# 15. Run experiments
# Config A & B already done — load from checkpoints.
# Config C & D: train fresh (edge_loss fix applied).
# ============================================================
NUM_EPOCHS = 100
results, histories = {}, {}

# ---- Load A & B results from saved JSON logs ----
for cfg, name in [('A', 'A_MCANet_Baseline'), ('B', 'B_MCANet_WFM_Only')]:
    log_path = RESULTS_DIR / f'{name}_log.json'
    if log_path.exists():
        data = json.loads(log_path.read_text())
        results[cfg]   = data['final']
        histories[cfg] = data['history']
        log(f'Loaded saved results for Config {cfg}: '
            f'Dice={results[cfg]["dice"]:.4f}  IoU={results[cfg]["iou"]:.4f}  '
            f'HD95={results[cfg]["hd95"]:.2f}px')
    else:
        log(f'WARNING: No saved log for Config {cfg} at {log_path}')

# ---- Config C — MCANet + Edge Loss only ----
model_C = MCANet(num_classes=2, dec_channels=64, num_heads=8, pretrained=True).to(DEVICE)
results['C'], histories['C'] = run_experiment(
    'C_MCANet_EdgeLoss_Only', model_C, use_edge_loss=True, num_epochs=NUM_EPOCHS)
torch.cuda.empty_cache()

# ---- Config D — WF-MCANet Full (our proposal) ----
model_D = WFMCANet(num_classes=2, dec_channels=64, num_heads=8, pretrained=True).to(DEVICE)
results['D'], histories['D'] = run_experiment(
    'D_WFMCANet_Full', model_D, use_edge_loss=True, num_epochs=NUM_EPOCHS)
torch.cuda.empty_cache()

# ============================================================
# 16. Cross-domain evaluation on Kvasir-SEG
# ============================================================
log('\n=== CROSS-DOMAIN EVALUATION ON KVASIR-SEG ===')
log('(Models trained on ISIC 2018, tested directly — no fine-tuning)')

ckpt_A = torch.load(CHECKPOINT_DIR / 'A_MCANet_Baseline_best.pt')
ckpt_D = torch.load(CHECKPOINT_DIR / 'D_WFMCANet_Full_best.pt')

eval_A = MCANet(num_classes=2, dec_channels=64, pretrained=False).to(DEVICE)
eval_D = WFMCANet(num_classes=2, dec_channels=64, pretrained=False).to(DEVICE)
eval_A.load_state_dict(ckpt_A['model'])
eval_D.load_state_dict(ckpt_D['model'])

kv_A = evaluate(eval_A, kvasir_loader)
kv_D = evaluate(eval_D, kvasir_loader)

log(f'MCANet  Baseline  — Dice={kv_A["dice"]:.4f}  IoU={kv_A["iou"]:.4f}  HD95={kv_A["hd95"]:.2f}px')
log(f'WF-MCANet (ours)  — Dice={kv_D["dice"]:.4f}  IoU={kv_D["iou"]:.4f}  HD95={kv_D["hd95"]:.2f}px')

# ============================================================
# 17. Training curves plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
cfgs = [
    ('A: Baseline',   histories['A'], '#000000', '-'),
    ('B: +WFM',       histories['B'], '#555555', '--'),
    ('C: +EdgeLoss',  histories['C'], '#888888', '-.'),
    ('D: WF-MCANet',  histories['D'], '#000000', ':'),
]
for ax, metric, ylabel in zip(axes,
                               ['dice', 'iou', 'hd95'],
                               ['Dice Score', 'IoU Score', 'HD95 (px)']):
    for name, h, color, ls in cfgs:
        if metric in h:
            ax.plot(h['epoch'], h[metric], label=name,
                    color=color, linestyle=ls, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs. Epoch')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
log('Training curves saved.')

# ============================================================
# 18. Final summary table + CSV
# ============================================================
log('\n' + '='*75)
log('FINAL RESULTS SUMMARY')
log('='*75)
log(f'{"Config":<35} {"Dice":>8} {"IoU":>8} {"HD95":>10} {"Params":>10}')
log('-'*75)

summary = [
    ('A: MCANet Baseline',           results['A'], 4.04),
    ('B: MCANet + WFM only',         results['B'], 4.24),
    ('C: MCANet + Edge Loss only',   results['C'], 4.04),
    ('D: WF-MCANet (full,proposed)', results['D'], 4.24),
]
for name, res, par in summary:
    log(f'{name:<35} {res["dice"]:>8.4f} {res["iou"]:>8.4f} '
        f'{res["hd95"]:>10.2f} {par:>8.2f}M')
log('-'*75)
log(f'{"WA-NET (ISIC 2018, published)":<35} {0.9458:>8.4f} {"---":>8} {"---":>10} {11.60:>8.2f}M')
log('='*75)
log(f'\nCross-domain (Kvasir-SEG, zero-shot):')
log(f'  MCANet : Dice={kv_A["dice"]:.4f}  IoU={kv_A["iou"]:.4f}  HD95={kv_A["hd95"]:.2f}px')
log(f'  Ours   : Dice={kv_D["dice"]:.4f}  IoU={kv_D["iou"]:.4f}  HD95={kv_D["hd95"]:.2f}px')

rows = [{'config': n, 'dice': r['dice'], 'iou': r['iou'],
         'hd95': r['hd95'], 'params_M': p} for n, r, p in summary]
pd.DataFrame(rows).to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
log(f'\nCSV saved: {RESULTS_DIR / "ablation_results.csv"}')

# ============================================================
# 19. WFM gate value
# ============================================================
gate_val = torch.sigmoid(eval_D.wfm.gate).item()
log(f'\nWFM gate after training: sigma(g) = {gate_val:.4f}')
if gate_val > 0.3:
    log('Wavelet features contributed meaningfully (gate > 0.3) ✓')
else:
    log('WARNING: gate low — wavelet contribution limited.')

log(f'\n=== Training complete. Results in: {RESULTS_DIR} ===')
log(f'    Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}')