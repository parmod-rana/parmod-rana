"""Self-contained R3 teacher optimization core.

This module deliberately has no dependency on legacy data.py/training.py so the
heavy trainer cannot accidentally import positional-shift targets or split-fitted
regime logic. All future supervision arrives only through frozen R3 target tensors.
"""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed: int = 20260905) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pinball(pred, target, qs=(0.1, 0.5, 0.9)):
    t = target.unsqueeze(-1)
    q = torch.tensor(qs, device=pred.device, dtype=pred.dtype).view(1, 1, 3)
    e = t - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def teacher_loss(out, b):
    """Frozen multi-task R3 teacher loss; labels are never model inputs."""
    loss = F.cross_entropy(out['action_logits'], b['action'])
    loss += 0.30 * F.cross_entropy(out['regime_logits'], b['regime'])
    loss += 0.035 * pinball(out['return_quantiles'], b['ret'])
    target_exc = torch.stack([b['mfe'], b['mae']], dim=-1)
    loss += 0.025 * F.smooth_l1_loss(out['excursions'], target_exc)
    raw = F.binary_cross_entropy_with_logits(
        out['target_before_stop_logits'], b['barrier'], reduction='none'
    )
    m = b['barrier_valid'].to(raw.dtype)
    denom = m.sum().clamp_min(1.0)
    loss += 0.25 * (raw * m).sum() / denom
    err = (out['return_quantiles'][:, 2, 1] - b['ret'][:, 2]).abs().detach().unsqueeze(-1)
    loss += 0.015 * F.smooth_l1_loss(out['uncertainty'], err)
    return loss
