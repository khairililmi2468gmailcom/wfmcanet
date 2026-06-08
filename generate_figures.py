#!/usr/bin/env python3
# =============================================================================
#  WF-MCANet — generate REAL journal figures + REAL metrics from trained models
# =============================================================================
#  Produces (in --out):
#    real_metrics.csv / real_metrics.txt   <- true params + Dice/IoU/HD95 (ISIC & Kvasir)
#    fig1_dataset_samples.png              <- ISIC + Kvasir samples (image | GT | overlay)
#    fig5_isic_qualitative.png             <- image | GT | baseline | WF-MCANet | overlay
#    fig7_kvasir_qualitative.png           <- zero-shot: image | GT | prediction | overlay
#    fig_feature_maps.png                  <- decoder feature + Haar LL/LH/HL/HH + conv-vs-wavelet
#    fig_training_curves.png               <- re-rendered from results/*_log.json (if present)
#
#  Model code is copied verbatim from train.py so checkpoints load exactly.
#  Nothing is fabricated: every number printed comes from running your model.
# =============================================================================
import os, json, math, argparse, glob, warnings
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------- backbone
import timm
_CANDIDATES = [
    ("mscan_t", (0,1,2,3)), ("mscan_s", (0,1,2,3)),
    ("mit_b1", (0,1,2,3)), ("mit_b0", (0,1,2,3)),
    ("efficientnet_b3", (1,2,3,4)), ("resnet50", (1,2,3,4)),
]
_avail = set(timm.list_models())
_BACKBONE, _IDX = None, None
for _n, _i in _CANDIDATES:
    if _n in _avail:
        try:
            _m = timm.create_model(_n, pretrained=False, features_only=True, out_indices=_i); del _m
            _BACKBONE, _IDX = _n, _i; break
        except Exception:
            continue
assert _BACKBONE, "No backbone found — install timm>=0.9.12"
print(f"[backbone] selected: {_BACKBONE}")

# ----------------------------------------------------------------------------- model (verbatim from train.py)
class MSCANEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(_BACKBONE, pretrained=pretrained,
                                          features_only=True, out_indices=_IDX)
        self.channels = self.backbone.feature_info.channels()
        self.backbone_name = _BACKBONE
    def forward(self, x): return self.backbone(x)

class DWConv1D(nn.Module):
    def __init__(self, dim, k, axis="x"):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, (1,k) if axis=="x" else (k,1),
                              padding=(0,k//2) if axis=="x" else (k//2,0), groups=dim)
    def forward(self, x): return self.conv(x)

class MultiScaleAxisConv(nn.Module):
    def __init__(self, dim, axis="x"):
        super().__init__()
        self.norm=nn.LayerNorm(dim); self.conv7=DWConv1D(dim,7,axis)
        self.conv11=DWConv1D(dim,11,axis); self.conv21=DWConv1D(dim,21,axis)
        self.proj=nn.Conv2d(dim,dim,1)
    def forward(self,x):
        xn=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.proj(self.conv7(xn)+self.conv11(xn)+self.conv21(xn))

class CrossAxisAttention(nn.Module):
    def __init__(self, dim, num_heads=8, axis="y"):
        super().__init__()
        self.num_heads=num_heads; self.head_dim=dim//num_heads
        self.scale=self.head_dim**-0.5; self.axis=axis
        self.to_q=nn.Linear(dim,dim,bias=False); self.to_kv=nn.Linear(dim,dim*2,bias=False)
        self.out=nn.Linear(dim,dim,bias=False)
    def forward(self, query_src, kv_src):
        B,C,H,W=query_src.shape; nh,hd=self.num_heads,self.head_dim
        if self.axis=="y":
            q=query_src.permute(0,3,2,1).reshape(B*W,H,C); kv=kv_src.permute(0,3,2,1).reshape(B*W,H,C); seq=H
        else:
            q=query_src.permute(0,2,3,1).reshape(B*H,W,C); kv=kv_src.permute(0,2,3,1).reshape(B*H,W,C); seq=W
        Q=self.to_q(q).reshape(-1,seq,nh,hd).transpose(1,2)
        KV=self.to_kv(kv).reshape(-1,seq,2,nh,hd).permute(2,0,3,1,4); K,V=KV[0],KV[1]
        attn=F.softmax((Q@K.transpose(-2,-1))*self.scale,dim=-1)
        out=self.out((attn@V).transpose(1,2).reshape(-1,seq,C))
        return out.reshape(B,W,H,C).permute(0,3,2,1) if self.axis=="y" else out.reshape(B,H,W,C).permute(0,3,1,2)

class MCABlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.x_conv=MultiScaleAxisConv(dim,"x"); self.y_conv=MultiScaleAxisConv(dim,"y")
        self.top_ca=CrossAxisAttention(dim,num_heads,"y"); self.bot_ca=CrossAxisAttention(dim,num_heads,"x")
        self.proj_t=nn.Conv2d(dim,dim,1); self.proj_b=nn.Conv2d(dim,dim,1)
    def forward(self,x):
        Fx=self.x_conv(x); Fy=self.y_conv(x)
        return self.proj_t(self.top_ca(Fy,Fx))+self.proj_b(self.bot_ca(Fx,Fy))+x

class WaveletFrequencyModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        haar=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],
                          dtype=torch.float32).view(4,1,2,2)*0.5
        self.register_buffer("haar",haar); self.channels=channels
        inner=max(channels*4//reduction,8)
        self.ca_fc1=nn.Linear(channels*4,inner); self.ca_fc2=nn.Linear(inner,channels*4)
        self.proj=nn.Sequential(nn.Conv2d(channels*4,channels,1,bias=False),nn.BatchNorm2d(channels))
        self.gate=nn.Parameter(torch.tensor(0.5))
    def forward(self,x):
        B,C,H,W=x.shape
        sub=F.conv2d(x.reshape(B*C,1,H,W),self.haar,stride=2).reshape(B,C*4,H//2,W//2)
        gap=sub.mean(dim=[2,3]); w=torch.sigmoid(self.ca_fc2(F.relu(self.ca_fc1(gap))))
        sub=sub*w.unsqueeze(-1).unsqueeze(-1); fhat=self.proj(sub)
        fhat=F.interpolate(fhat,size=(H,W),mode="bilinear",align_corners=False)
        return x+torch.sigmoid(self.gate)*fhat

def _proj(i,o): return nn.Sequential(nn.Conv2d(i,o,1),nn.BatchNorm2d(o),nn.ReLU())

class MCANet(nn.Module):
    def __init__(self, num_classes=2, dec_channels=64, num_heads=8, pretrained=False):
        super().__init__()
        self.encoder=MSCANEncoder(pretrained); ec=self.encoder.channels
        self.up_e2=_proj(ec[1],dec_channels); self.up_e3=_proj(ec[2],dec_channels); self.up_e4=_proj(ec[3],dec_channels)
        self.reduce_conv=_proj(dec_channels*3,dec_channels); self.mca=MCABlock(dec_channels,num_heads)
        self.fuse_e1=nn.Sequential(_proj(dec_channels+ec[0],dec_channels),
                                   nn.Conv2d(dec_channels,dec_channels,1),nn.BatchNorm2d(dec_channels),nn.ReLU())
        self.head=nn.Conv2d(dec_channels,num_classes,1)
    def forward(self,x):
        H,W=x.shape[2:]; E1,E2,E3,E4=self.encoder(x); ts=E1.shape[2:]
        e2=F.interpolate(self.up_e2(E2),ts,mode="bilinear",align_corners=False)
        e3=F.interpolate(self.up_e3(E3),ts,mode="bilinear",align_corners=False)
        e4=F.interpolate(self.up_e4(E4),ts,mode="bilinear",align_corners=False)
        fused=self.reduce_conv(torch.cat([e2,e3,e4],1)); attn=self.mca(fused)
        out=self.fuse_e1(torch.cat([attn,E1],1))
        return self.head(F.interpolate(out,(H,W),mode="bilinear",align_corners=False))

class WFMCANet(nn.Module):
    def __init__(self, num_classes=2, dec_channels=64, num_heads=8, pretrained=False):
        super().__init__()
        self.encoder=MSCANEncoder(pretrained); ec=self.encoder.channels
        self.up_e2=_proj(ec[1],dec_channels); self.up_e3=_proj(ec[2],dec_channels); self.up_e4=_proj(ec[3],dec_channels)
        self.reduce_conv=_proj(dec_channels*3,dec_channels)
        self.wfm=WaveletFrequencyModule(dec_channels,4); self.mca=MCABlock(dec_channels,num_heads)
        self.fuse_e1=nn.Sequential(_proj(dec_channels+ec[0],dec_channels),
                                   nn.Conv2d(dec_channels,dec_channels,1),nn.BatchNorm2d(dec_channels),nn.ReLU())
        self.head=nn.Conv2d(dec_channels,num_classes,1)
    def forward(self,x):
        H,W=x.shape[2:]; E1,E2,E3,E4=self.encoder(x); ts=E1.shape[2:]
        e2=F.interpolate(self.up_e2(E2),ts,mode="bilinear",align_corners=False)
        e3=F.interpolate(self.up_e3(E3),ts,mode="bilinear",align_corners=False)
        e4=F.interpolate(self.up_e4(E4),ts,mode="bilinear",align_corners=False)
        fused=self.reduce_conv(torch.cat([e2,e3,e4],1)); fused_wf=self.wfm(fused); attn=self.mca(fused_wf)
        out=self.fuse_e1(torch.cat([attn,E1],1))
        return self.head(F.interpolate(out,(H,W),mode="bilinear",align_corners=False))

# config_name -> model class (A/C = MCANet, B/D = WFMCANet)
def model_for(name):
    return WFMCANet() if ("WFM" in name or "Full" in name) else MCANet()

def load_ckpt(path):
    name=Path(path).stem.replace("_best","")
    model=model_for(name)
    ckpt=torch.load(path, map_location="cpu", weights_only=False)
    state=ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    state={k.replace("module.",""):v for k,v in state.items()}
    msd=model.state_dict()
    matched={k:v for k,v in state.items() if k in msd and v.shape==msd[k].shape}
    model.load_state_dict(matched, strict=False)
    pct=100*len(matched)/len(msd)
    total=sum(p.numel() for p in model.parameters())
    print(f"[load] {name}: {len(matched)}/{len(msd)} keys ({pct:.0f}%) | params={total/1e6:.2f}M | backbone={_BACKBONE}")
    return model.to(DEVICE).eval(), total/1e6, pct

# ----------------------------------------------------------------------------- data
def isic_pairs(data_dir):
    imgs=Path(data_dir)/"ISIC2018"/"ISIC2018_Task1-2_Training_Input"
    msks=Path(data_dir)/"ISIC2018"/"ISIC2018_Task1_Training_GroundTruth"
    if not imgs.is_dir():
        for c in Path(data_dir).rglob("ISIC2018_Task1-2_Training_Input"):
            imgs=c; msks=c.parent/"ISIC2018_Task1_Training_GroundTruth"; break
    allp=[(str(p), str(msks/(p.stem+"_segmentation.png"))) for p in sorted(imgs.glob("*.jpg"))]
    allp=[(i,m) for i,m in allp if os.path.exists(m)]
    split=Path(data_dir)/"isic2018_split.json"
    if not split.exists():
        for c in Path(data_dir).rglob("isic2018_split.json"): split=c; break
    if split.exists():
        try:
            s=json.load(open(split)); test=s.get("test")
            if test and isinstance(test[0],(list,tuple)):
                print(f"[isic] HELD-OUT test split from {Path(split).name}: {len(test)} imgs"); return [tuple(x) for x in test]
            if test:
                tset=set(Path(t).stem.replace('_segmentation','') for t in test)
                pairs=[(i,m) for i,m in allp if Path(i).stem in tset]
                print(f"[isic] HELD-OUT test split from {Path(split).name}: {len(pairs)} imgs"); return pairs
        except Exception as e:
            print("[isic] split json parse failed:", e)
    n=len(allp); fb=allp[int(0.8*n):]
    print(f"[isic] *** WARNING: isic2018_split.json NOT found. Using fallback last-20% ({len(fb)} imgs).")
    print( "[isic] *** These may OVERLAP training images and INFLATE Dice. Copy your real")
    print( "[isic] *** isic2018_split.json (created by train.py) to the data dir and re-run.")
    return fb

def kvasir_pairs(data_dir):
    root=Path(data_dir)
    best=None
    for imgs in root.rglob("images"):
        msks=imgs.parent/"masks"
        if not msks.is_dir(): continue
        pairs=[(str(p),str(msks/p.name)) for p in sorted(imgs.glob("*.jpg")) if (msks/p.name).exists()]
        if not pairs:
            pairs=[(str(p),str(msks/(p.stem+'.png'))) for p in sorted(imgs.glob("*.jpg")) if (msks/(p.stem+'.png')).exists()]
        if pairs and (best is None or 'kvasir' in str(imgs).lower()):
            best=(str(imgs),pairs)
    if best:
        print(f"[kvasir] using {best[0]} ({len(best[1])} pairs)"); return best[1]
    print(f"[kvasir] *** WARNING: no images/masks folder found under {root}.")
    print( "[kvasir] *** Kvasir-SEG was NOT downloaded on this server — that is why its")
    print( "[kvasir] *** metrics came back NaN. Run setup.sh (Kvasir section) first.")
    return []

def load_image(path):
    arr=np.array(Image.open(path).convert("RGB").resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR),np.float32)/255.
    return arr  # H,W,3 in [0,1]

def load_mask(path):
    m=np.array(Image.open(path).convert("L").resize((IMG_SIZE,IMG_SIZE),Image.NEAREST))
    return (m>127).astype(np.uint8)

def to_tensor(img01):
    arr=(img01-MEAN)/STD
    return torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)

@torch.no_grad()
def predict(model, img01):
    logits=model(to_tensor(img01))
    return torch.softmax(logits,1)[0,1].cpu().numpy()  # H,W prob

# ----------------------------------------------------------------------------- metrics
try:
    from medpy.metric.binary import hd95 as _hd95
except Exception:
    _hd95=None

def metrics(prob, gt, thr=0.5):
    p=(prob>=thr).astype(np.uint8); t=gt.astype(np.uint8); sm=1e-5
    inter=(p&t).sum(); dice=(2*inter+sm)/(p.sum()+t.sum()+sm); iou=(inter+sm)/((p|t).sum()+sm)
    hd=float("nan")
    if _hd95 is not None and p.sum()>0 and t.sum()>0:
        try: hd=float(_hd95(p,t))
        except Exception: pass
    return float(dice), float(iou), hd

@torch.no_grad()
def evaluate(model, pairs, max_n=None):
    ds, ious, hds = [], [], []
    pl=pairs if max_n is None else pairs[:max_n]
    for i,(ip,mp) in enumerate(pl):
        prob=predict(model, load_image(ip)); d,j,h=metrics(prob, load_mask(mp))
        ds.append(d); ious.append(j)
        if not math.isnan(h): hds.append(h)
        if (i+1)%50==0: print(f"    eval {i+1}/{len(pl)}")
    return (float(np.mean(ds)), float(np.std(ds)),
            float(np.mean(ious)), float(np.mean(hds)) if hds else float("nan"),
            float(np.std(hds)) if hds else float("nan"))

# ----------------------------------------------------------------------------- figure helpers
def overlay(img01, mask, color=(0,0.82,0.43), alpha=0.42):
    o=img01.copy()
    for c in range(3): o[:,:,c]=o[:,:,c]*(1-alpha*mask)+color[c]*alpha*mask
    # red boundary
    from scipy.ndimage import binary_erosion
    b=mask.astype(bool)&~binary_erosion(mask.astype(bool),iterations=2)
    o[b]=[1,0.23,0.23]
    return np.clip(o,0,1)

def panel(ax, im, title, cmap=None):
    ax.imshow(im, cmap=cmap); ax.set_title(title, fontsize=9); ax.axis("off")

def fig_dataset_samples(out, isic, kvasir, n=2):
    rows=[("ISIC 2018",isic[:n]),("Kvasir-SEG",kvasir[:n])]
    fig,axs=plt.subplots(2*n,3,figsize=(7.2,2.4*n*2)); axs=np.array(axs).reshape(2*n,3)
    r=0
    for dname,pairs in rows:
        for ip,mp in pairs:
            im=load_image(ip); gt=load_mask(mp)
            panel(axs[r,0],im,f"{dname}: image"); panel(axs[r,1],gt,"ground truth",cmap="gray")
            panel(axs[r,2],overlay(im,gt),"mask overlay"); r+=1
    plt.tight_layout(); fig.savefig(out/"fig1_dataset_samples.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    print("  saved fig1_dataset_samples.png")

def fig_qualitative(out, pairs, baseline, ours, fname, title, n=3):
    cols=["Image","Ground Truth"]+(["Baseline (MCANet)"] if baseline else [])+["WF-MCANet (Ours)","Overlay (Ours)"]
    fig,axs=plt.subplots(n,len(cols),figsize=(2.2*len(cols),2.4*n)); axs=np.array(axs).reshape(n,len(cols))
    for r,(ip,mp) in enumerate(pairs[:n]):
        im=load_image(ip); gt=load_mask(mp); c=0
        panel(axs[r,c],im,cols[c] if r==0 else ""); c+=1
        panel(axs[r,c],gt,cols[c] if r==0 else "",cmap="gray"); c+=1
        if baseline:
            pb=(predict(baseline,im)>=0.5).astype(np.uint8); panel(axs[r,c],pb,cols[c] if r==0 else "",cmap="gray"); c+=1
        po=predict(ours,im); pm=(po>=0.5).astype(np.uint8)
        panel(axs[r,c],pm,cols[c] if r==0 else "",cmap="gray"); c+=1
        panel(axs[r,c],overlay(im,pm),cols[c] if r==0 else "")
    plt.suptitle(title,fontsize=10,y=1.005); plt.tight_layout()
    fig.savefig(out/fname,dpi=200,bbox_inches="tight"); plt.close(fig)
    print(f"  saved {fname}")

def fig_feature_maps(out, ours, sample_img):
    """Extract the decoder feature 'fused' (input to the WFM) via a hook, then
       show its Haar LL/LH/HL/HH sub-bands and a conventional-conv (Laplacian)
       comparison — illustrates what spatial conv misses vs the frequency domain."""
    feats={}
    h=ours.reduce_conv.register_forward_hook(lambda m,i,o: feats.__setitem__("f",o.detach()))
    _=predict(ours, sample_img); h.remove()
    f=feats["f"]                                   # 1,C,h,w
    fm=f.mean(1,keepdim=True)                       # 1,1,h,w mean activation
    haar=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],
                      dtype=torch.float32).view(4,1,2,2).to(f.device)*0.5
    sub=F.conv2d(fm, haar, stride=2)[0].cpu().numpy()   # 4,h/2,w/2
    fmn=fm[0,0].cpu().numpy()
    lap=np.abs(F.conv2d(fm, torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32
            ).view(1,1,3,3).to(f.device), padding=1)[0,0].cpu().numpy())
    names=["Decoder feature (mean)","LL (Low-Low)","LH (Low-High)","HL (High-Low)",
           "HH (High-High, diagonal)","Conventional conv (Laplacian)"]
    imgs=[fmn, sub[0], sub[1], sub[2], sub[3], lap]
    fig,axs=plt.subplots(1,7,figsize=(17,2.7))
    axs[0].imshow(sample_img); axs[0].set_title("Input image",fontsize=8.5); axs[0].axis("off")
    for k,(im,nm) in enumerate(zip(imgs,names),start=1):
        axs[k].imshow(im,cmap="inferno"); axs[k].set_title(nm,fontsize=8.5); axs[k].axis("off")
    plt.tight_layout(); fig.savefig(out/"fig_feature_maps.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    print("  saved fig_feature_maps.png")

def fig_training_curves(out, results_dir):
    logs=sorted(glob.glob(str(Path(results_dir)/"*_log.json")))
    if not logs: print("  (no *_log.json found — skipping training curves)"); return
    fig,axs=plt.subplots(1,3,figsize=(13,3.4))
    for lg in logs:
        d=json.load(open(lg)); hist=d.get("history",{}); name=d.get("config","")
        for ax,key,ylab in zip(axs,["dice","iou","hd95"],["Dice","IoU","HD95 (px)"]):
            y=hist.get(key,[])
            if y: ax.plot(range(1,len(y)+1), y, label=name); ax.set_title(ylab); ax.set_xlabel("eval step")
    axs[0].legend(fontsize=7)
    plt.tight_layout(); fig.savefig(out/"fig_training_curves.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("  saved fig_training_curves.png")

# ----------------------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="/mnt/gpu17/segilmi/checkpoints")
    ap.add_argument("--data", default="/mnt/gpu17/segilmi/data")
    ap.add_argument("--results", default="/mnt/gpu17/segilmi/results")
    ap.add_argument("--out", default="/mnt/gpu17/segilmi/journal_assets")
    ap.add_argument("--max-eval", type=int, default=None, help="cap #images for metric eval (None=all)")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)

    isic=isic_pairs(args.data); kvasir=kvasir_pairs(args.data)
    print(f"[data] ISIC test pairs={len(isic)} | Kvasir pairs={len(kvasir)} | device={DEVICE}")

    # locate checkpoints
    ck={}
    for tag in ["A_MCANet_Baseline","B_MCANet_WFM_Only","C_MCANet_EdgeLoss_Only","D_WFMCANet_Full"]:
        p=Path(args.checkpoints)/f"{tag}_best.pt"
        if p.exists(): ck[tag]=str(p)
    if not ck:
        # fall back: any *.pt in checkpoints dir
        for p in glob.glob(str(Path(args.checkpoints)/"*.pt")): ck[Path(p).stem.replace("_best","")]=p
    print(f"[ckpt] found: {list(ck)}")

    models={}; rows=[]
    for tag,path in ck.items():
        m,params,pct=load_ckpt(path); models[tag]=m
        if isic:
            print(f"  evaluating {tag} on ISIC ...")
            d,ds,j,h,hs=evaluate(m, isic, args.max_eval)
        else:
            d=ds=j=h=hs=float("nan")
        kd=kj=kh=float("nan")
        if kvasir:
            print(f"  evaluating {tag} on Kvasir (zero-shot) ...")
            kd,_,kj,kh,_=evaluate(m, kvasir, args.max_eval)
        rows.append(dict(config=tag, params_M=round(params,2), keys_loaded_pct=round(pct,1),
                         isic_dice=round(d,4), isic_dice_std=round(ds,4), isic_iou=round(j,4),
                         isic_hd95=round(h,2), isic_hd95_std=round(hs,2),
                         kvasir_dice=round(kd,4), kvasir_iou=round(kj,4), kvasir_hd95=round(kh,2)))

    # write real metrics
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out/"real_metrics.csv", index=False)
    except Exception:
        pass
    with open(out/"real_metrics.txt","w") as f:
        f.write(f"backbone = {_BACKBONE}\n\n")
        for r in rows:
            f.write(json.dumps(r)+"\n")
    print("\n===== REAL MEASURED RESULTS (use these in the paper) =====")
    print(f"backbone = {_BACKBONE}")
    for r in rows: print(r)
    print("==========================================================\n")

    # figures
    if isic and kvasir: fig_dataset_samples(out, isic, kvasir)
    baseline=models.get("A_MCANet_Baseline")
    ours=models.get("D_WFMCANet_Full") or models.get("B_MCANet_WFM_Only")
    if ours and isic:
        fig_qualitative(out, isic, baseline, ours, "fig5_isic_qualitative.png",
                        "Qualitative segmentation results on ISIC 2018", n=3)
        fig_feature_maps(out, ours, load_image(isic[0][0]))
    if ours and kvasir:
        fig_qualitative(out, kvasir, None, ours, "fig7_kvasir_qualitative.png",
                        "Zero-shot qualitative results on Kvasir-SEG", n=2)
    fig_training_curves(out, args.results)
    print(f"\nAll outputs written to: {out}")

if __name__=="__main__":
    main()