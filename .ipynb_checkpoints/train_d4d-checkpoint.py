#!/usr/bin/env python3
"""
WF-MCANet D4d — Resume dari D4b best checkpoint (Dice=0.9061, ep140)
Perbaikan MINIMAL yang aman:
  1. Resume dari D4b checkpoint (ep140, Dice=0.9061)
  2. Ganti CosineAnnealing → CosineAnnealingLR biasa (T_max=260, tanpa restart)
  3. LR lebih kecil: 5e-5 (fine-tuning dari model yang sudah bagus)
  4. Total 400 epoch (260 epoch tambahan dari ep140)
  5. Semua hal lain SAMA PERSIS dengan D4b (augment, gate, loss)
  Tidak ubah: augmentasi, gate, grad clip, arsitektur
"""
import os,sys,time,json,math,random,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
from PIL import Image
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
DEVICE=torch.device('cuda')

BASE_DIR=Path('/mnt/gpu17/segilmi'); DATA_DIR=BASE_DIR/'data'
KVASIR_DIR=DATA_DIR/'Kvasir-SEG'
RESULTS_DIR=BASE_DIR/'results'; CHECKPOINT_DIR=BASE_DIR/'checkpoints'
LOG_FILE=BASE_DIR/'train_d4d.log'
for d in [RESULTS_DIR,CHECKPOINT_DIR]: d.mkdir(parents=True,exist_ok=True)

class Tee:
    def __init__(self,p): self.f=open(p,'a',buffering=1); self.t=sys.stdout
    def write(self,m): self.t.write(m); self.f.write(m)
    def flush(self): self.t.flush(); self.f.flush()
sys.stdout=Tee(LOG_FILE)
def log(*a,**k): print(*a,**k); sys.stdout.flush()

log('='*65)
log('  WF-MCANet D4d — Resume D4b ep140 + CosLR no-restart + lr=5e-5')
log(f'  Started : {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'  GPU     : {torch.cuda.get_device_name(0)}')
log(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
log('='*65)

# ============================================================
# Arsitektur IDENTIK dengan D4b (tidak ada perubahan)
# ============================================================
def drop_path_fn(x,p=0.,tr=False):
    if p==0. or not tr: return x
    keep=1-p; s=(x.shape[0],)+(1,)*(x.ndim-1)
    return x.div(keep)*torch.floor(torch.rand(s,dtype=x.dtype,device=x.device)+keep)
class DropPath(nn.Module):
    def __init__(self,p=0.): super().__init__(); self.p=p
    def forward(self,x): return drop_path_fn(x,self.p,self.training)

class SpatialGatingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv0  =nn.Conv2d(dim,dim,5,padding=2,groups=dim)
        self.conv0_1=nn.Conv2d(dim,dim,(1,7),padding=(0,3),groups=dim)
        self.conv0_2=nn.Conv2d(dim,dim,(7,1),padding=(3,0),groups=dim)
        self.conv1_1=nn.Conv2d(dim,dim,(1,11),padding=(0,5),groups=dim)
        self.conv1_2=nn.Conv2d(dim,dim,(11,1),padding=(5,0),groups=dim)
        self.conv2_1=nn.Conv2d(dim,dim,(1,21),padding=(0,10),groups=dim)
        self.conv2_2=nn.Conv2d(dim,dim,(21,1),padding=(10,0),groups=dim)
        self.conv3  =nn.Conv2d(dim,dim,1)
    def forward(self,x):
        u=x.clone(); a=self.conv0(x)
        a=(self.conv0_1(a)+self.conv0_2(a)+self.conv1_1(a)+
           self.conv1_2(a)+self.conv2_1(a)+self.conv2_2(a))
        return u*self.conv3(a)

class MSCAAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.proj_1=nn.Conv2d(dim,dim,1); self.act=nn.GELU()
        self.spatial_gating_unit=SpatialGatingUnit(dim)
        self.proj_2=nn.Conv2d(dim,dim,1)
    def forward(self,x):
        sc=x; x=self.proj_1(x); x=self.act(x)
        x=self.spatial_gating_unit(x); x=self.proj_2(x)
        return x+sc

class MixFFN(nn.Module):
    def __init__(self,d,hd,drop=0.):
        super().__init__()
        self.fc1=nn.Conv2d(d,hd,1); self.act=nn.GELU()
        self.dwconv=nn.Conv2d(hd,hd,3,padding=1,groups=hd)
        self.norm=nn.BatchNorm2d(hd)
        self.fc2=nn.Conv2d(hd,d,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x)
        x=self.dwconv(x); x=self.norm(x); x=self.drop(x)
        x=self.fc2(x); x=self.drop(x); return x

class MSCANBlock(nn.Module):
    def __init__(self,dim,mlp_ratio=4.,drop=0.,dp=0.):
        super().__init__()
        self.norm1=nn.BatchNorm2d(dim); self.attn=MSCAAttention(dim)
        self.norm2=nn.BatchNorm2d(dim); self.mlp=MixFFN(dim,int(dim*mlp_ratio),drop=drop)
        self.dp=DropPath(dp) if dp>0. else nn.Identity()
        self.layer_scale_1=nn.Parameter(1e-2*torch.ones(dim))
        self.layer_scale_2=nn.Parameter(1e-2*torch.ones(dim))
    def forward(self,x):
        x=x+self.dp(self.layer_scale_1[:,None,None]*self.attn(self.norm1(x)))
        x=x+self.dp(self.layer_scale_2[:,None,None]*self.mlp(self.norm2(x)))
        return x

class StemConv(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.proj=nn.Sequential(
            nn.Conv2d(ic,oc//2,3,stride=2,padding=1,bias=True),nn.BatchNorm2d(oc//2),nn.GELU(),
            nn.Conv2d(oc//2,oc,3,stride=2,padding=1,bias=True),nn.BatchNorm2d(oc))
    def forward(self,x): return self.proj(x)

class OPEmbed(nn.Module):
    def __init__(self,ic,oc,stride=2):
        super().__init__()
        self.proj=nn.Conv2d(ic,oc,3,stride=stride,padding=1,bias=True)
        self.norm=nn.BatchNorm2d(oc)
    def forward(self,x): return self.norm(self.proj(x))

class MSCAN(nn.Module):
    MLP_RATIOS=[8,8,4,4]
    def __init__(self,embed_dims=[32,64,160,256],depths=[3,3,5,2],
                 drop_rate=0.,drop_path_rate=0.1,pretrained=None):
        super().__init__()
        mlpr=self.MLP_RATIOS
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]
        cur=0
        for i in range(4):
            ic=3 if i==0 else embed_dims[i-1]; oc=embed_dims[i]
            emb=StemConv(ic,oc) if i==0 else OPEmbed(ic,oc,stride=2)
            blk=nn.ModuleList([MSCANBlock(oc,mlp_ratio=mlpr[i],
                drop=drop_rate,dp=dpr[cur+j]) for j in range(depths[i])])
            norm=nn.BatchNorm2d(oc)
            setattr(self,f'patch_embed{i+1}',emb)
            setattr(self,f'block{i+1}',blk)
            setattr(self,f'norm{i+1}',norm); cur+=depths[i]
        if pretrained: self._load(pretrained)
    def _load(self,path):
        import argparse
        try: torch.serialization.add_safe_globals([argparse.Namespace])
        except: pass
        w=torch.load(path,map_location='cpu',weights_only=False)
        sd=w.get('state_dict',w.get('model',w))
        sd={k:v for k,v in sd.items() if not k.startswith('head.')}
        msd=self.state_dict()
        ok={k:v for k,v in sd.items() if k in msd and v.shape==msd[k].shape}
        self.load_state_dict(ok,strict=False)
        log(f'  MSCAN-T pretrained: {len(ok)}/{len(sd)} loaded')
    def forward(self,x):
        outs=[]
        for i in range(4):
            x=getattr(self,f'patch_embed{i+1}')(x)
            for b in getattr(self,f'block{i+1}'): x=b(x)
            x=getattr(self,f'norm{i+1}')(x); outs.append(x)
        return outs

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
        self.gate=nn.Parameter(torch.tensor(0.1))
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
        self.c=nn.Sequential(nn.Conv2d(c,c//2,3,padding=1),nn.BatchNorm2d(c//2),
                              nn.ReLU(),nn.Conv2d(c//2,1,1))
    def forward(self,x): return self.c(x)
def pr(ic,oc):
    return nn.Sequential(nn.Conv2d(ic,oc,1),nn.BatchNorm2d(oc),nn.ReLU())

class WFMCANet(nn.Module):
    def __init__(self,dc=64,h=8,pretrained=False):
        super().__init__()
        # pretrained=False karena kita akan load dari D4b checkpoint
        self.backbone=MSCAN(embed_dims=[32,64,160,256],depths=[3,3,5,2],
            drop_path_rate=0.1,pretrained=None)
        ec=[32,64,160,256]
        self.u2=pr(ec[1],dc); self.u3=pr(ec[2],dc); self.u4=pr(ec[3],dc)
        self.red=pr(dc*3,dc); self.wfm=MultiScaleWFM(dc); self.mca=MCABlock(dc,h)
        self.fuse=nn.Sequential(pr(dc+ec[0],dc),nn.Conv2d(dc,dc,1),nn.BatchNorm2d(dc),nn.ReLU())
        self.head=nn.Conv2d(dc,1,1); self.bnd=BndHead(dc)
    def forward(self,x):
        H,W=x.shape[2:]
        E1,E2,E3,E4=self.backbone(x); ts=E1.shape[2:]
        e2=F.interpolate(self.u2(E2),ts,mode='bilinear',align_corners=False)
        e3=F.interpolate(self.u3(E3),ts,mode='bilinear',align_corners=False)
        e4=F.interpolate(self.u4(E4),ts,mode='bilinear',align_corners=False)
        f=self.red(torch.cat([e2,e3,e4],1)); f=self.wfm(f); f=self.mca(f)
        d=self.fuse(torch.cat([f,E1],1))
        o=F.interpolate(d,(H,W),mode='bilinear',align_corners=False)
        if self.training: return self.head(o),self.bnd(o)
        return self.head(o)

# ============================================================
# Loss — IDENTIK dengan D4b
# ============================================================
def bce_loss(logit,target):
    return F.binary_cross_entropy_with_logits(logit.squeeze(1).float(),target.float())
def dice_loss_fg(logit,target,smooth=1.0):
    prob=torch.sigmoid(logit.squeeze(1)); tgt=target.float()
    pf=prob.reshape(prob.shape[0],-1); tf=tgt.reshape(tgt.shape[0],-1)
    inter=(pf*tf).sum(1); denom=pf.sum(1)+tf.sum(1)
    return 1.-((2.*inter+smooth)/(denom+smooth)).mean()
def edge_loss(bnd,target,w=0.1):
    sx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,
                    device=target.device).view(1,1,3,3)
    mf=target.float().unsqueeze(1)
    edge=(F.conv2d(mf,sx,padding=1).abs()+F.conv2d(mf,sx.transpose(2,3),padding=1).abs()).clamp(0,1)
    return w*F.binary_cross_entropy_with_logits(bnd.float(),edge)
def total_loss(seg,bnd,t):
    return 0.5*bce_loss(seg,t)+0.5*dice_loss_fg(seg,t)+edge_loss(bnd,t)

from medpy.metric.binary import hd95 as _hd95
def metrics(logit,masks):
    preds=(torch.sigmoid(logit.squeeze(1))>0.5).long()
    dl,il,hl=[],[],[]; sm=1e-5
    for p,t in zip(preds,masks):
        p=p.cpu().numpy().astype(bool); t=t.cpu().numpy().astype(bool)
        i=(p&t).sum()
        dl.append((2*i+sm)/(p.sum()+t.sum()+sm))
        il.append((i+sm)/((p|t).sum()+sm))
        if p.sum()>0 and t.sum()>0:
            try: hl.append(_hd95(p,t))
            except: pass
    return {'dice':float(np.mean(dl)),'iou':float(np.mean(il)),
            'hd95':float(np.mean(hl)) if hl else float('nan')}

# ============================================================
# Dataset — IDENTIK dengan D4b (Resize biasa, bukan RandomCrop)
# ============================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMG_SIZE=512; BATCH=32; NW=0

tr_t=A.Compose([
    A.Resize(IMG_SIZE,IMG_SIZE),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Rotate(limit=30,p=0.5),
    A.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.05,p=0.5),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()
])
va_t=A.Compose([
    A.Resize(IMG_SIZE,IMG_SIZE),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()
])

class Seg(Dataset):
    def __init__(self,pairs,t=None): self.p=pairs; self.t=t
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        im,mk=self.p[i]
        im=np.array(Image.open(im).convert('RGB'))
        mk=(np.array(Image.open(mk).convert('L'))>127).astype(np.uint8)
        if self.t:
            a=self.t(image=im,mask=mk); im,mk=a['image'],a['mask'].long()
        return im,mk

sp=json.loads((DATA_DIR/'isic2018_split.json').read_text())
kv_img=KVASIR_DIR/'images'; kv_msk=KVASIR_DIR/'masks'
kv_pairs=[(str(i),str(kv_msk/i.name)) for i in sorted(kv_img.glob('*.jpg'))
          if (kv_msk/i.name).exists()]
tr_ld=DataLoader(Seg(sp['train'],tr_t),batch_size=BATCH,shuffle=True,num_workers=NW,pin_memory=False)
te_ld=DataLoader(Seg(sp['test'],va_t),batch_size=8,shuffle=False,num_workers=NW,pin_memory=False)
kv_ld=DataLoader(Seg(kv_pairs,va_t),batch_size=8,shuffle=False,num_workers=NW,pin_memory=False)
log(f'Train:{len(tr_ld)}b  Test:{len(te_ld)}b  Kvasir:{len(kv_ld)}b  Batch:{BATCH}')

# ============================================================
# Training
# ============================================================
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
                lg=torch.logit((torch.sigmoid(lg)+torch.sigmoid(lh)+torch.sigmoid(lv))/3.,eps=1e-6)
        torch.cuda.synchronize(); tt+=time.time()-t0; n+=imgs.shape[0]
        m=metrics(lg,masks); dl.append(m['dice']); il.append(m['iou'])
        if not math.isnan(m['hd95']): hl.append(m['hd95'])
    return {'dice':float(np.mean(dl)),'iou':float(np.mean(il)),
            'hd95':float(np.mean(hl)) if hl else float('nan')}

def run_resume(name, ckpt_path, extra_epochs=260, lr=5e-5, wd=1e-4, ev=5):
    log(f'\n{"="*65}')
    log(f'  {name}')
    log(f'  Resume dari: {ckpt_path}')
    log(f'  Extra epochs: {extra_epochs} | LR: {lr} | Scheduler: CosineAnnealingLR')
    log(f'  Augment: IDENTIK D4b (Resize+flip+rotate+jitter)')
    log(f'{"="*65}')

    # Load checkpoint D4b
    model = WFMCANet(dc=64,h=8,pretrained=False).to(DEVICE)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model'])
    start_dice = sd['metrics']['dice']
    log(f'  Loaded checkpoint: Dice={start_dice:.4f} @ ep{sd["epoch"]}')
    log(f'  WFM gate saat resume: {model.wfm.gate.item():.4f}')

    ck = CHECKPOINT_DIR/f'{name}_best.pt'
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9,0.999))
    # CosineAnnealingLR: decay smooth dari lr ke eta_min, TANPA restart
    sch = CosineAnnealingLR(opt, T_max=extra_epochs, eta_min=1e-6)
    sc = torch.amp.GradScaler('cuda')
    best = start_dice; hist = defaultdict(list); t0 = time.time()

    for ep in range(1, extra_epochs+1):
        tl = train_ep(model, tr_ld, opt, sc)
        sch.step()

        do = (ep%ev==0) or ep==extra_epochs or ep==1
        if do:
            m = evaluate(model, te_ld)
            hist['epoch'].append(ep); hist['loss'].append(tl)
            hist['dice'].append(m['dice']); hist['iou'].append(m['iou'])
            star = ''
            if m['dice'] > best:
                best = m['dice']
                torch.save({'epoch':sd['epoch']+ep,'model':model.state_dict(),'metrics':m}, ck)
                star = '  <-- best'
            hd = f"{m['hd95']:.2f}" if not math.isnan(m['hd95']) else ' nan'
            cur_lr = opt.param_groups[0]['lr']
            gate_v = model.wfm.gate.item()
            log(f'Ep+{ep:3d} (abs={sd["epoch"]+ep}) | loss={tl:.4f} | '
                f'Dice={m["dice"]:.4f} | IoU={m["iou"]:.4f} | '
                f'HD95={hd}px | lr={cur_lr:.1e} | gate={gate_v:.3f}{star}')
        else:
            ela=time.time()-t0; eta=int(ela*(extra_epochs-ep)/max(ep,1)/60)
            log(f'Ep+{ep:3d} (abs={sd["epoch"]+ep}) | loss={tl:.4f} | ETA~{eta}min')
        t0 = time.time()

    # Final eval
    final_sd = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_state_dict(final_sd['model'])
    fm = evaluate(model, te_ld)
    log(f'\n{"-"*65}')
    log(f'  FINAL [{name}]')
    log(f'  Start: Dice={start_dice:.4f} (D4b ep140)')
    log(f'  Final: Dice={fm["dice"]:.4f}  IoU={fm["iou"]:.4f}  HD95={fm["hd95"]:.2f}px')
    log(f'  Delta: {fm["dice"]-start_dice:+.4f}')
    log(f'{"-"*65}')
    json.dump({'history':dict(hist),'final':fm},
              open(RESULTS_DIR/f'{name}_log.json','w'),indent=2)
    return fm, model

# ============================================================
# Main
# ============================================================
D4B_CKPT = CHECKPOINT_DIR / 'D4b_WFMCANet_fixed_best.pt'
if not D4B_CKPT.exists():
    log(f'ERROR: Checkpoint D4b tidak ditemukan: {D4B_CKPT}')
    log('Pastikan path benar. List checkpoints:')
    for f in sorted(CHECKPOINT_DIR.glob('*.pt')):
        log(f'  {f}')
    sys.exit(1)

log(f'\n>>> D4d: Resume D4b (Dice=0.9061) + 260 epoch tambahan <<<')
results, model = run_resume('D4d_WFMCANet', D4B_CKPT, extra_epochs=260, lr=5e-5)

log('\n=== CROSS-DOMAIN: KVASIR-SEG ===')
kv = evaluate(model, kv_ld, tta=False)
log(f'Kvasir: Dice={kv["dice"]:.4f}  IoU={kv["iou"]:.4f}  HD95={kv["hd95"]:.2f}px')

log('\n'+'='*65)
rows=[
    ('MCANet-T (paper)',      0.9293, 0.9040),
    ('D4b (ep140, baseline)', 0.9061, 0.8407),
    ('D4d (resume+260ep)',    results['dice'], results['iou']),
]
log(f'{"Method":<30} {"Dice":>7} {"IoU":>7}')
log('-'*45)
for nm,d,i in rows:
    log(f'{nm:<30} {d:>7.4f} {i:>7.4f}')
log('='*65)
log(f'\nDone: {time.strftime("%Y-%m-%d %H:%M:%S")}')