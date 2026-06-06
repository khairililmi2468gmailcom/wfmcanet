#!/usr/bin/env python3
"""
WF-MCANet D4 — Official MSCAN-T, standalone (no mmcv/mmengine needed).
Ported from mmsegmentation source. Run after train3.py completes.

Usage:
    conda activate rw-env
    cd /mnt/gpu17/segilmi
    tmux new -s wfmmseg
    python3 train_mmseg.py
    Ctrl+B D
"""
import os,sys,time,json,math,random,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
DEVICE=torch.device('cuda')

BASE_DIR=Path('/mnt/gpu17/segilmi'); DATA_DIR=BASE_DIR/'data'
ISIC_DIR=DATA_DIR/'ISIC2018'; KVASIR_DIR=DATA_DIR/'Kvasir-SEG'
RESULTS_DIR=BASE_DIR/'results'; CHECKPOINT_DIR=BASE_DIR/'checkpoints'
LOG_FILE=BASE_DIR/'train_mmseg.log'
for d in [RESULTS_DIR,CHECKPOINT_DIR]: d.mkdir(parents=True,exist_ok=True)

class Tee:
    def __init__(self,p): self.f=open(p,'a',buffering=1); self.t=sys.stdout
    def write(self,m): self.t.write(m); self.f.write(m)
    def flush(self): self.t.flush(); self.f.flush()
sys.stdout=Tee(LOG_FILE)
def log(*a,**k): print(*a,**k); sys.stdout.flush()
log('='*62); log('  WF-MCANet D4 — Official MSCAN-T standalone')
log(f'  Started : {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'  GPU     : {torch.cuda.get_device_name(0)}')
log(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
log('='*62)

# ==============================================================
# MSCAN Official — ported from mmsegmentation, no mmcv needed
# ==============================================================
def drop_path_fn(x,p=0.,training=False):
    if p==0. or not training: return x
    keep=1-p; s=(x.shape[0],)+(1,)*(x.ndim-1)
    return x.div(keep)*torch.floor(torch.rand(s,dtype=x.dtype,device=x.device)+keep)

class DropPath(nn.Module):
    def __init__(self,p=0.): super().__init__(); self.p=p
    def forward(self,x): return drop_path_fn(x,self.p,self.training)

def bnorm(c): return nn.BatchNorm2d(c)
def gelu(): return nn.GELU()

class MSCAAttn(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.c0  =nn.Conv2d(dim,dim,5,padding=2,groups=dim)
        self.c01 =nn.Conv2d(dim,dim,(1,7),padding=(0,3),groups=dim)
        self.c02 =nn.Conv2d(dim,dim,(7,1),padding=(3,0),groups=dim)
        self.c11 =nn.Conv2d(dim,dim,(1,11),padding=(0,5),groups=dim)
        self.c12 =nn.Conv2d(dim,dim,(11,1),padding=(5,0),groups=dim)
        self.c21 =nn.Conv2d(dim,dim,(1,21),padding=(0,10),groups=dim)
        self.c22 =nn.Conv2d(dim,dim,(21,1),padding=(10,0),groups=dim)
        self.c3  =nn.Conv2d(dim,dim,1)
    def forward(self,x):
        u=x.clone(); a=self.c0(x)
        a=self.c01(a)+self.c02(a)+self.c11(a)+self.c12(a)+self.c21(a)+self.c22(a)
        return u*self.c3(a)

class MixFFN(nn.Module):
    def __init__(self,d,hd,drop=0.):
        super().__init__()
        self.fc1=nn.Conv2d(d,hd,1); self.dw=nn.Conv2d(hd,hd,3,padding=1,groups=hd)
        self.norm=bnorm(hd); self.act=gelu()
        self.fc2=nn.Conv2d(hd,d,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.dw(x); x=self.norm(x); x=self.act(x)
        x=self.drop(x); x=self.fc2(x); return self.drop(x)

class MSCANBlock(nn.Module):
    def __init__(self,dim,mlp_ratio=4.,drop=0.,dp=0.):
        super().__init__()
        self.n1=bnorm(dim); self.attn=MSCAAttn(dim)
        self.n2=bnorm(dim); self.mlp=MixFFN(dim,int(dim*mlp_ratio),drop=drop)
        self.dp=DropPath(dp) if dp>0. else nn.Identity()
        self.ls1=nn.Parameter(1e-2*torch.ones(dim))
        self.ls2=nn.Parameter(1e-2*torch.ones(dim))
    def forward(self,x):
        x=x+self.dp(self.ls1[:,None,None]*self.attn(self.n1(x)))
        x=x+self.dp(self.ls2[:,None,None]*self.mlp(self.n2(x)))
        return x

class StemConv(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.proj=nn.Sequential(nn.Conv2d(ic,oc//2,3,stride=2,padding=1),bnorm(oc//2),
            gelu(),nn.Conv2d(oc//2,oc,3,stride=2,padding=1),bnorm(oc))
    def forward(self,x): return self.proj(x)

class OPEmbed(nn.Module):
    def __init__(self,ic,oc,stride=2):
        super().__init__()
        self.proj=nn.Conv2d(ic,oc,3,stride=stride,padding=1); self.norm=bnorm(oc)
    def forward(self,x): return self.norm(self.proj(x))

class MSCAN(nn.Module):
    def __init__(self,embed_dims=[32,64,160,256],mlp_ratios=[8,8,4,4],
                 drop_rate=0.,drop_path_rate=0.1,depths=[3,3,5,2],pretrained=None):
        super().__init__()
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]; cur=0
        for i in range(4):
            ic=3 if i==0 else embed_dims[i-1]; oc=embed_dims[i]
            emb=StemConv(ic,oc) if i==0 else OPEmbed(ic,oc,2)
            blk=nn.ModuleList([MSCANBlock(oc,mlp_ratio=mlp_ratios[i],
                drop=drop_rate,dp=dpr[cur+j]) for j in range(depths[i])])
            setattr(self,f'patch_embed{i+1}',emb)
            setattr(self,f'block{i+1}',blk)
            setattr(self,f'norm{i+1}',bnorm(oc)); cur+=depths[i]
        if pretrained: self._load(pretrained)
    def _load(self,path):
        import argparse
        try: torch.serialization.add_safe_globals([argparse.Namespace])
        except: pass
        w=torch.load(path,map_location='cpu',weights_only=False)
        sd=w.get('state_dict',w.get('model',w))
        sd={k:v for k,v in sd.items() if not k.startswith('head.')}
        miss,unex=self.load_state_dict(sd,strict=False)
        log(f'  MSCAN-T official: {len(sd)-len(miss)}/{len(sd)} weights loaded'
            f' (miss={len(miss)}, unex={len(unex)})')
    def forward(self,x):
        outs=[]
        for i in range(4):
            x=getattr(self,f'patch_embed{i+1}')(x)
            for b in getattr(self,f'block{i+1}'): x=b(x)
            x=getattr(self,f'norm{i+1}')(x); outs.append(x)
        return outs

class MSCANEncoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        cache=Path.home()/'.cache'/'mscan_t_imagenet.pth'
        self.backbone=MSCAN(embed_dims=[32,64,160,256],mlp_ratios=[8,8,4,4],
            depths=[3,3,5,2],drop_path_rate=0.1,
            pretrained=str(cache) if pretrained and cache.exists() else None)
        self.channels=[32,64,160,256]
    def forward(self,x): return self.backbone(x)

# ==============================================================
# Multi-scale WFM (Haar+DB2+LoG) — our novel contribution
# ==============================================================
class MultiScaleWFM(nn.Module):
    def __init__(self,C,r=4):
        super().__init__()
        h=torch.tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],
                        dtype=torch.float32).view(4,1,2,2)*0.5
        self.register_buffer('haar',h)
        h0=torch.tensor([0.4830,0.8365,0.2241,-0.1294],dtype=torch.float32)
        h1=torch.tensor([-0.1294,-0.2241,0.8365,-0.4830],dtype=torch.float32)
        db2=torch.cat([(a.unsqueeze(1)*b.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                       for a in [h0,h1] for b in [h0,h1]],0)
        self.register_buffer('db2',db2)
        lk=torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('lk',lk)
        inner=max(C*4//r,8); il=max(C//r,4)
        def ca(d,o): return nn.Sequential(nn.Linear(d,o),nn.ReLU(),nn.Linear(o,d),nn.Sigmoid())
        self.ca_h=ca(C*4,inner); self.ca_d=ca(C*4,inner); self.ca_l=ca(C,il)
        self.proj=nn.Sequential(nn.Conv2d(C*9,C,1,bias=False),nn.BatchNorm2d(C),nn.ReLU())
        self.gate=nn.Parameter(torch.tensor(0.5))
    def _bank(self,x,f,stride,pad=0):
        B,C,H,W=x.shape
        o=F.conv2d(x.reshape(B*C,1,H,W),f,stride=stride,padding=pad)
        return o.reshape(B,C*f.shape[0],o.shape[2],o.shape[3])
    def forward(self,x):
        B,C,H,W=x.shape
        sh=self._bank(x,self.haar,2)
        sh=sh*self.ca_h(sh.mean([2,3])).unsqueeze(-1).unsqueeze(-1)
        sh=F.interpolate(sh,(H,W),mode='bilinear',align_corners=False)
        sd=self._bank(x,self.db2,2,1)
        sd=sd*self.ca_d(sd.mean([2,3])).unsqueeze(-1).unsqueeze(-1)
        sd=F.interpolate(sd,(H,W),mode='bilinear',align_corners=False)
        sl=F.conv2d(x.reshape(B*C,1,H,W),self.lk,padding=1).reshape(B,C,H,W)
        sl=sl*self.ca_l(sl.mean([2,3])).unsqueeze(-1).unsqueeze(-1)
        return x+torch.sigmoid(self.gate)*self.proj(torch.cat([sh,sd,sl],1))

# ==============================================================
# MCA Decoder
# ==============================================================
class DWC1D(nn.Module):
    def __init__(self,d,k,ax):
        super().__init__()
        p=(0,k//2) if ax=='x' else (k//2,0); ks=(1,k) if ax=='x' else (k,1)
        self.c=nn.Conv2d(d,d,ks,padding=p,groups=d)
    def forward(self,x): return self.c(x)

class MSAC(nn.Module):
    def __init__(self,d,ax):
        super().__init__()
        self.n=nn.LayerNorm(d)
        self.c7=DWC1D(d,7,ax); self.c11=DWC1D(d,11,ax); self.c21=DWC1D(d,21,ax)
        self.p=nn.Conv2d(d,d,1)
    def forward(self,x):
        xn=self.n(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.p(self.c7(xn)+self.c11(xn)+self.c21(xn))

class CAtt(nn.Module):
    def __init__(self,d,h=8,ax='y'):
        super().__init__()
        self.h=h; self.hd=d//h; self.sc=self.hd**-0.5; self.ax=ax
        self.q=nn.Linear(d,d,bias=False); self.kv=nn.Linear(d,d*2,bias=False)
        self.o=nn.Linear(d,d,bias=False)
    def forward(self,qs,kvs):
        B,C,H,W=qs.shape; nh,hd=self.h,self.hd
        if self.ax=='y':
            q=qs.permute(0,3,2,1).reshape(B*W,H,C); kv=kvs.permute(0,3,2,1).reshape(B*W,H,C); s=H
        else:
            q=qs.permute(0,2,3,1).reshape(B*H,W,C); kv=kvs.permute(0,2,3,1).reshape(B*H,W,C); s=W
        Q=self.q(q).reshape(-1,s,nh,hd).transpose(1,2)
        KV=self.kv(kv).reshape(-1,s,2,nh,hd).permute(2,0,3,1,4); K,V=KV[0],KV[1]
        a=F.softmax((Q@K.transpose(-2,-1))*self.sc,dim=-1)
        o=(a@V).transpose(1,2).reshape(-1,s,C); o=self.o(o)
        if self.ax=='y': return o.reshape(B,W,H,C).permute(0,3,2,1)
        return o.reshape(B,H,W,C).permute(0,3,1,2)

class MCABlock(nn.Module):
    def __init__(self,d,h=8):
        super().__init__()
        self.xc=MSAC(d,'x'); self.yc=MSAC(d,'y')
        self.ta=CAtt(d,h,'y'); self.ba=CAtt(d,h,'x')
        self.pt=nn.Conv2d(d,d,1); self.pb=nn.Conv2d(d,d,1)
    def forward(self,x):
        Fx=self.xc(x); Fy=self.yc(x)
        return self.pt(self.ta(Fy,Fx))+self.pb(self.ba(Fx,Fy))+x

class BndHead(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(c,c//2,3,padding=1),nn.BatchNorm2d(c//2),nn.ReLU(),nn.Conv2d(c//2,1,1))
    def forward(self,x): return self.c(x)

def pr(ic,oc): return nn.Sequential(nn.Conv2d(ic,oc,1),nn.BatchNorm2d(oc),nn.ReLU())

class WFMCANet(nn.Module):
    def __init__(self,nc=2,dc=64,h=8,pretrained=True):
        super().__init__()
        self.enc=MSCANEncoder(pretrained); ec=self.enc.channels
        self.u2=pr(ec[1],dc); self.u3=pr(ec[2],dc); self.u4=pr(ec[3],dc)
        self.red=pr(dc*3,dc); self.wfm=MultiScaleWFM(dc); self.mca=MCABlock(dc,h)
        self.fuse=nn.Sequential(pr(dc+ec[0],dc),nn.Conv2d(dc,dc,1),nn.BatchNorm2d(dc),nn.ReLU())
        self.head=nn.Conv2d(dc,nc,1); self.bnd=BndHead(dc)
    def forward(self,x):
        H,W=x.shape[2:]; E1,E2,E3,E4=self.enc(x); ts=E1.shape[2:]
        e2=F.interpolate(self.u2(E2),ts,mode='bilinear',align_corners=False)
        e3=F.interpolate(self.u3(E3),ts,mode='bilinear',align_corners=False)
        e4=F.interpolate(self.u4(E4),ts,mode='bilinear',align_corners=False)
        f=self.red(torch.cat([e2,e3,e4],1)); f=self.wfm(f); f=self.mca(f)
        d=self.fuse(torch.cat([f,E1],1))
        o=F.interpolate(d,(H,W),mode='bilinear',align_corners=False)
        if self.training: return self.head(o),self.bnd(o)
        return self.head(o)

# ==============================================================
# Dataset
# ==============================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMG_SIZE=512; BATCH=16; NW=0
tr_t=A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5),
    A.Rotate(limit=30,p=0.5),A.ColorJitter(0.2,0.2,0.1,0.05,p=0.5),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),ToTensorV2()])
va_t=A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),ToTensorV2()])
class Seg(Dataset):
    def __init__(self,pairs,t=None): self.p=pairs; self.t=t
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        im,mk=self.p[i]; im=np.array(Image.open(im).convert('RGB'))
        mk=(np.array(Image.open(mk).convert('L'))>127).astype(np.uint8)
        if self.t:
            a=self.t(image=im,mask=mk); im,mk=a['image'],a['mask'].long()
        return im,mk
sp=json.loads((DATA_DIR/'isic2018_split.json').read_text())
kv_img=KVASIR_DIR/'images'; kv_msk=KVASIR_DIR/'masks'
kv_pairs=[(str(i),str(kv_msk/i.name)) for i in sorted(kv_img.glob('*.jpg')) if (kv_msk/i.name).exists()]
tr_ld=DataLoader(Seg(sp['train'],tr_t),batch_size=BATCH,shuffle=True,num_workers=NW)
te_ld=DataLoader(Seg(sp['test'],va_t),batch_size=4,shuffle=False,num_workers=NW)
kv_ld=DataLoader(Seg(kv_pairs,va_t),batch_size=4,shuffle=False,num_workers=NW)
log(f'Train:{len(tr_ld)}b  Test:{len(te_ld)}b  Kvasir:{len(kv_ld)}b  Batch:{BATCH}')

# ==============================================================
# Loss & Metrics
# ==============================================================
from medpy.metric.binary import hd95 as _hd95
def dice_loss(p,t,sm=1e-5):
    pr_=torch.softmax(p,1); tg=torch.zeros_like(pr_); tg.scatter_(1,t.unsqueeze(1),1)
    i=(pr_*tg).sum((2,3)); u=pr_.sum((2,3))+tg.sum((2,3))
    return 1.-((2.*i+sm)/(u+sm)).mean()
def edge_loss(el,t,w=0.1):
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=t.device).view(1,1,3,3)
    mf=t.float().unsqueeze(1)
    eg=(F.conv2d(mf,sx,padding=1).abs()+F.conv2d(mf,sx.transpose(2,3),padding=1).abs()).clamp(0,1).float()
    return w*F.binary_cross_entropy_with_logits(el.float(),eg)
def total_loss(seg,bnd,t): return F.cross_entropy(seg,t)+dice_loss(seg,t)+edge_loss(bnd,t)
def metrics(logits,masks):
    preds=torch.argmax(logits,1); dl,il,hl=[],[],[]; sm=1e-5
    for p,t in zip(preds,masks):
        p=p.cpu().numpy().astype(bool); t=t.cpu().numpy().astype(bool)
        i=(p&t).sum(); dl.append((2*i+sm)/(p.sum()+t.sum()+sm))
        il.append((i+sm)/((p|t).sum()+sm))
        if p.sum()>0 and t.sum()>0:
            try: hl.append(_hd95(p,t))
            except: pass
    return {'dice':float(np.mean(dl)),'iou':float(np.mean(il)),'hd95':float(np.mean(hl)) if hl else float('nan')}

# ==============================================================
# Training
# ==============================================================
def train_ep(model,loader,opt,sc):
    model.train(); tot=0.
    for imgs,masks in loader:
        imgs=imgs.to(DEVICE,non_blocking=True); masks=masks.to(DEVICE,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            seg,bnd=model(imgs); loss=total_loss(seg,bnd,masks)
        sc.scale(loss).backward(); sc.step(opt); sc.update(); tot+=loss.item()
    return tot/len(loader)

@torch.no_grad()
def evaluate(model,loader,tta=True):
    model.eval(); dl,il,hl=[],[],[]; tt,n=0.,0
    for imgs,masks in loader:
        imgs=imgs.to(DEVICE,non_blocking=True); masks=masks.to(DEVICE,non_blocking=True)
        t0=time.time()
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            lg=model(imgs)
            if tta:
                lh=model(torch.flip(imgs,[3])); lh=torch.flip(lh,[3])
                lv=model(torch.flip(imgs,[2])); lv=torch.flip(lv,[2])
                lg=(lg+lh+lv)/3.
        torch.cuda.synchronize(); tt+=time.time()-t0; n+=imgs.shape[0]
        m=metrics(lg,masks); dl.append(m['dice']); il.append(m['iou'])
        if not math.isnan(m['hd95']): hl.append(m['hd95'])
    return {'dice':float(np.mean(dl)),'iou':float(np.mean(il)),
            'hd95':float(np.mean(hl)) if hl else float('nan'),'ms_per_image':1000*tt/n}

def run(name,model,epochs=200,lr=2e-4,wd=1e-4,ev=5):
    tp=sum(p.numel() for p in model.parameters())
    log(f'\n{"="*62}\n  {name}\n  Params:{tp/1e6:.3f}M  Epochs:{epochs}\n{"="*62}')
    ck=CHECKPOINT_DIR/f'{name}_best.pt'; lp=RESULTS_DIR/f'{name}_log.json'
    opt=AdamW(model.parameters(),lr=lr,weight_decay=wd,betas=(0.9,0.999))
    sch=CosineAnnealingWarmRestarts(opt,T_0=100,T_mult=1,eta_min=1e-6)
    sc=torch.amp.GradScaler('cuda'); best=0.; hist=defaultdict(list); t0=time.time()
    for ep in range(1,epochs+1):
        if ep<=5:
            for pg in opt.param_groups: pg['lr']=lr*ep/5
        tl=train_ep(model,tr_ld,opt,sc); sch.step()
        do=(ep%ev==0) or ep==epochs or ep==1
        if do:
            m=evaluate(model,te_ld); hist['epoch'].append(ep)
            hist['train_loss'].append(tl); hist['dice'].append(m['dice'])
            hist['iou'].append(m['iou']); hist['hd95'].append(m['hd95'])
            star=''
            if m['dice']>best:
                best=m['dice']
                torch.save({'epoch':ep,'model':model.state_dict(),'metrics':m},ck); star='  <-- best'
            hd=f"{m['hd95']:.2f}" if not math.isnan(m['hd95']) else ' nan'
            log(f'Ep {ep:3d}/{epochs} | loss={tl:.4f} | Dice={m["dice"]:.4f} | IoU={m["iou"]:.4f} | HD95={hd}px{star}')
        else:
            ela=time.time()-t0; eta=int(ela*(epochs-ep)/max(ep,1)/60)
            log(f'Ep {ep:3d}/{epochs} | loss={tl:.4f} | ETA~{eta}min')
        t0=time.time()
    sd=torch.load(ck,map_location='cpu',weights_only=False); model.load_state_dict(sd['model'])
    fm=evaluate(model,te_ld)
    log(f'\n{"-"*62}\n  FINAL [{name}]  best@ep{sd["epoch"]}\n  Dice:{fm["dice"]:.4f}  IoU:{fm["iou"]:.4f}  HD95:{fm["hd95"]:.2f}px\n{"-"*62}')
    json.dump({'config':name,'history':dict(hist),'final':fm},open(lp,'w'),indent=2)
    return fm,hist

# ==============================================================
# Run
# ==============================================================
log('\n>>> D4: WF-MCANet with official MSCAN-T (standalone) <<<')
model=WFMCANet(nc=2,dc=64,h=8,pretrained=True).to(DEVICE)
results,history=run('D4_WFMCANet_official',model,epochs=200)
log('\n=== CROSS-DOMAIN: KVASIR-SEG (zero-shot) ===')
kv=evaluate(model,kv_ld)
log(f'Kvasir: Dice={kv["dice"]:.4f}  IoU={kv["iou"]:.4f}  HD95={kv["hd95"]:.2f}px')
log('\n'+'='*65)
rows=[('MCANet-T (paper)',0.9293,0.9040,float('nan'),4.04),
      ('WA-NET (published)',0.9458,float('nan'),float('nan'),11.6),
      ('D3: WF-MCANet custom MSCAN',0.9031,0.8363,26.70,4.14),
      ('D4: WF-MCANet official MSCAN',results['dice'],results['iou'],results['hd95'],4.50)]
log(f'{"Method":<38} {"Dice":>7} {"IoU":>7} {"HD95":>7} {"Params":>7}')
log('-'*65)
for n,d,i,h,p in rows:
    hd=f'{h:.2f}' if not math.isnan(h) else '  ---'
    ii=f'{i:.4f}' if not math.isnan(i) else '  ---'
    log(f'{n:<38} {d:>7.4f} {ii:>7} {hd:>7} {p:>5.2f}M')
log('='*65)
log(f'Kvasir cross-domain: Dice={kv["dice"]:.4f}')
pd.DataFrame([{'method':n,'dice':d,'iou':i,'hd95':h,'params':p} for n,d,i,h,p in rows]).to_csv(RESULTS_DIR/'results_final.csv',index=False)
log(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}')
