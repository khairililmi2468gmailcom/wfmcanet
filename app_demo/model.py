"""
WF-MCANet Demo Model — IDENTIK dengan train.py (Config B: B_MCANet_WFM_Only)

Checkpoint: B_MCANet_WFM_Only_best.pt (40MB)
Backbone  : EfficientNet-B3 via timm (features_only, out_indices=(1,2,3,4))
            channels = [32, 48, 136, 384]
WFM       : WaveletFrequencyModule — Haar only, gate=Parameter(0.5)
Decoder   : MCABlock (dc=64, 8 heads)
Output    : 2-class logits → argmax → foreground mask
Key prefx : encoder.backbone.* / up_e2.* / up_e3.* / up_e4.* /
            reduce_conv.* / wfm.* / mca.* / fuse_e1.* / head.*
"""

import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path


# ── Encoder (timm backbone wrapper) ──────────────────────────────────────────
class MSCANEncoder(nn.Module):
    """
    Wrapper identik dengan train.py MSCANEncoder.
    Mencoba backbone dari list prioritas; EfficientNet-B3 untuk Config B.
    key: encoder.backbone.*
    """
    CANDIDATES = [
        ('mscan_t',         (0,1,2,3)),
        ('efficientnet_b3', (1,2,3,4)),
        ('resnet50',        (1,2,3,4)),
    ]

    def __init__(self, pretrained=False):
        super().__init__()
        import timm
        chosen = None
        available = set(timm.list_models())
        for name, idx in self.CANDIDATES:
            if name not in available:
                continue
            try:
                m = timm.create_model(name, pretrained=False,
                                      features_only=True, out_indices=idx)
                chosen = (name, idx, m)
                break
            except Exception:
                continue
        if chosen is None:
            raise RuntimeError("No suitable backbone found. Install timm.")
        self.backbone_name = chosen[0]
        self.backbone = chosen[2]
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)   # returns [E1, E2, E3, E4]


# ── MCA Decoder (identik train.py) ───────────────────────────────────────────
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
        xn = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.proj(self.conv7(xn) + self.conv11(xn) + self.conv21(xn))

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
            return out.reshape(B, W, H, C).permute(0,3,2,1)
        return out.reshape(B, H, W, C).permute(0,3,1,2)

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
        return self.proj_t(self.top_ca(Fy,Fx)) + self.proj_b(self.bot_ca(Fx,Fy)) + x


# ── WaveletFrequencyModule — IDENTIK train.py ─────────────────────────────────
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
        # pad jika H atau W ganjil
        if H % 2: x = F.pad(x, (0,0,0,1))
        if W % 2: x = F.pad(x, (0,1,0,0))
        sub  = F.conv2d(x.reshape(B*C, 1, x.shape[2], x.shape[3]),
                        self.haar, stride=2).reshape(B, C*4,
                        x.shape[2]//2, x.shape[3]//2)
        gap  = sub.mean(dim=[2,3])
        w    = torch.sigmoid(self.ca_fc2(F.relu(self.ca_fc1(gap))))
        sub  = sub * w.unsqueeze(-1).unsqueeze(-1)
        fhat = self.proj(sub)
        fhat = F.interpolate(fhat, size=(H, W), mode='bilinear', align_corners=False)
        return x[:,:,:H,:W] + torch.sigmoid(self.gate) * fhat


# ── Full WFMCANet — IDENTIK train.py ─────────────────────────────────────────
def _make_proj(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class WFMCANet(nn.Module):
    """
    IDENTIK dengan class WFMCANet di train.py.
    Key structure:
      encoder.backbone.*   (EfficientNet-B3 ~566 keys)
      up_e2.* up_e3.* up_e4.*
      reduce_conv.*
      wfm.*
      mca.*
      fuse_e1.*
      head.*
    Output: (B, 2, H, W) logits  →  foreground = argmax == 1
    """
    def __init__(self, dec_channels=64, num_heads=8):
        super().__init__()
        self.encoder     = MSCANEncoder(pretrained=False)
        ec               = self.encoder.channels
        self.up_e2       = _make_proj(ec[1], dec_channels)
        self.up_e3       = _make_proj(ec[2], dec_channels)
        self.up_e4       = _make_proj(ec[3], dec_channels)
        self.reduce_conv = _make_proj(dec_channels * 3, dec_channels)
        self.wfm         = WaveletFrequencyModule(dec_channels, reduction=4)
        self.mca         = MCABlock(dec_channels, num_heads)
        self.fuse_e1     = nn.Sequential(
            _make_proj(dec_channels + ec[0], dec_channels),
            nn.Conv2d(dec_channels, dec_channels, 1),
            nn.BatchNorm2d(dec_channels), nn.ReLU()
        )
        self.head = nn.Conv2d(dec_channels, 2, 1)

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
        return self.head(out)   # (B, 2, H, W) — 2-class logits


def load_model(checkpoint_path=None, device='cpu'):
    """Load WFMCANet dan checkpoint. Kompatibel dengan B_MCANet_WFM_Only_best.pt."""
    model = WFMCANet(dec_channels=64, num_heads=8)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # train.py saves: {'epoch': ep, 'model': model.state_dict(), 'metrics': m}
        state = ckpt.get('model', ckpt.get('model_state_dict',
                         ckpt.get('state_dict', ckpt)))
        # Strip DataParallel prefix jika ada
        state = {k.replace('module.', ''): v for k, v in state.items()}

        msd     = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in msd and v.shape == msd[k].shape}
        n_miss  = len(msd) - len(matched)
        n_unex  = len([k for k in state if k not in msd])

        model.load_state_dict(matched, strict=False)
        pct = len(matched) / len(msd) * 100
        print(f'[model] Loaded {len(matched)}/{len(msd)} keys ({pct:.0f}%) '
              f'from {Path(checkpoint_path).name}')
        if n_miss:
            print(f'  Missing (random init): {n_miss} keys')
        if n_unex:
            print(f'  Unexpected (ignored): {n_unex} keys')
    else:
        print('[model] No checkpoint — random init')

    model.to(device).eval()
    p = sum(p.numel() for p in model.parameters()) / 1e6
    bn = getattr(model.encoder, 'backbone_name', 'unknown')
    ch = model.encoder.channels
    print(f'[model] {p:.2f} M params | backbone={bn} channels={ch} | device={device}')
    return model


if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    m    = load_model(ckpt)
    x    = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = m(x)
    # output: (1,2,512,512) — foreground prob via softmax[:,1]
    prob = torch.softmax(out, dim=1)[0, 1].numpy()
    print(f'Output logits: {out.shape}')
    print(f'Foreground prob: min={prob.min():.3f} max={prob.max():.3f} mean={prob.mean():.3f}')
