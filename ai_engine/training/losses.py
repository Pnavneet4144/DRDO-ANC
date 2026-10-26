"""Scale-invariant SNR loss for speech enhancement training."""

from __future__ import annotations

import torch


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR in dB: [B, T] -> [B]."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s_ref_sq = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target = dot * ref / s_ref_sq
    e_noise = est - s_target
    si = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(si + eps)


def si_snr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Negative mean SI-SNR — lower is better."""
    return -si_snr(est, ref).mean()
