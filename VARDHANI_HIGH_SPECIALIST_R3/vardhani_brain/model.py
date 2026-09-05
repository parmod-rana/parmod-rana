from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import BrainConfig

class CausalBlock(nn.Module):
    def __init__(self,h,heads,ff_mult,dropout):
        super().__init__()
        self.ln1=nn.LayerNorm(h); self.attn=nn.MultiheadAttention(h,heads,dropout=dropout,batch_first=True)
        self.ln2=nn.LayerNorm(h)
        self.ff=nn.Sequential(nn.Linear(h,h*ff_mult),nn.GELU(),nn.Dropout(dropout),nn.Linear(h*ff_mult,h),nn.Dropout(dropout))
    def forward(self,x):
        t=x.size(1); mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        y=self.ln1(x); y,_=self.attn(y,y,y,attn_mask=mask,need_weights=False)
        x=x+y; x=x+self.ff(self.ln2(x)); return x

class Specialist(nn.Module):
    def __init__(self,h,eh,dropout):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,eh),nn.GELU(),nn.Dropout(dropout),nn.Linear(eh,h))
    def forward(self,x): return self.net(x)

class MarketBrain(nn.Module):
    specialist_names=(
        "trend_continuation", "reversal", "range_chop", "breakout_breakdown",
        "failed_break_path_transition", "multi_timeframe_conflict", "volatility_vix",
        "ce_pe_options", "oi_volume_liquidity", "cross_index", "mfe_mae_risk",
        "no_trade_uncertainty"
    )
    def __init__(self,cfg:BrainConfig):
        super().__init__(); self.seq_len=cfg.seq_len; self.num_horizons=len(cfg.horizons)
        self.lower_in=nn.Linear(cfg.lower_dim,cfg.hidden_dim)
        self.ctx15_in=nn.Linear(cfg.context15_dim,cfg.hidden_dim)
        self.pos=nn.Parameter(torch.zeros(1,cfg.seq_len,cfg.hidden_dim))
        self.blocks=nn.ModuleList([CausalBlock(cfg.hidden_dim,cfg.heads,cfg.ff_mult,cfg.dropout) for _ in range(cfg.layers)])
        self.final_ln=nn.LayerNorm(cfg.hidden_dim)
        ne=max(cfg.experts,10)
        self.experts=nn.ModuleList([Specialist(cfg.hidden_dim,cfg.expert_hidden,cfg.dropout) for _ in range(ne)])
        self.gate=nn.Linear(cfg.hidden_dim,ne)
        self.latent=nn.Linear(cfg.hidden_dim,cfg.latent_dim)
        # Action path uses lower-timeframe branch only. 15m is context/reporting, never a mechanical veto.
        self.action=nn.Linear(cfg.hidden_dim,cfg.actions)
        self.regime=nn.Linear(cfg.hidden_dim,cfg.regimes)
        self.ret=nn.Linear(cfg.hidden_dim,len(cfg.horizons)*3) # q10/q50/q90 bps
        self.excursion=nn.Linear(cfg.hidden_dim,len(cfg.horizons)*2) # MFE/MAE
        self.barrier=nn.Linear(cfg.hidden_dim,len(cfg.horizons))
        self.uncertainty=nn.Linear(cfg.hidden_dim,1)
        self.ctx15=nn.Sequential(nn.LayerNorm(cfg.hidden_dim),nn.Linear(cfg.hidden_dim,64),nn.GELU(),nn.Linear(64,4))
    def forward(self,lower,ctx15):
        # lower: [B,T,lower_dim], ctx15: [B,T,context15_dim]
        t=lower.size(1)
        x=self.lower_in(lower)+self.pos[:,:t]
        for b in self.blocks: x=b(x)
        x=self.final_ln(x)
        h=x[:,-1]
        g=torch.softmax(self.gate(h),-1)
        ex=torch.stack([e(h) for e in self.experts],dim=1)
        fused=h+(g.unsqueeze(-1)*ex).sum(1)
        c=self.ctx15_in(ctx15[:,-1])
        return {
            "action_logits":self.action(fused),
            "regime_logits":self.regime(fused),
            "return_quantiles":self.ret(fused).view(-1,self.num_horizons,3),
            "excursions":self.excursion(fused).view(-1,self.num_horizons,2),
            "target_before_stop_logits":self.barrier(fused),
            "uncertainty":F.softplus(self.uncertainty(fused)),
            "latent":F.normalize(self.latent(fused),dim=-1),
            "expert_weights":g,
            "context15":self.ctx15(c)
        }

def parameter_report(model):
    n=sum(p.numel() for p in model.parameters())
    return {"parameters":n,"fp32_bytes":n*4,"fp16_bytes":n*2,"int8_bytes":n}
