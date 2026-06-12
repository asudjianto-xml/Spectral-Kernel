"""The spectral-mixture front-end -- H frequency banks fused into one kernel.

A single bank has one length scale: ``k = exp(-||phi(x) - phi(x')|| / T)``. Adding
frequencies to that bank samples one spectral density more finely; it cannot add
scales. A spectral MIXTURE fuses H banks, each with its own frequency band, its own
embedding and its own bandwidth, summed convexly:

    K(x,x') = sum_h w_h * exp(-||phi_h(x) - phi_h(x')|| / T_h)

The RKHS of a sum of kernels is the direct sum of the per-bank RKHSs, so smooth
global structure (a large-T_h bank) and sharp local detail (a small-T_h bank) are
represented simultaneously, each at its own scale and weight. With learned w_h and
T_h this is a learnable spectral-mixture kernel (Wilson & Adams). ``H=1`` recovers
the single bank.

The banks share the ARD relevance and the encoder; only the frequencies, amplitudes
and bandwidth are per-bank (per-bank encoders overfit -- a finding from the lineage).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import init_omega


def init_banks(d, K, H, omega_range, seed, device, dtype):
    """Initial frequency grids for H banks. H=1 spans the whole range; H>1 splits the
    log-frequency axis into H overlapping bands so the banks start at different scales."""
    if H == 1:
        return [init_omega(d, K, omega_range, seed, device, dtype)]
    lo, hi = omega_range
    edges = lo * (hi / lo) ** (np.arange(H + 1) / H)
    rng = np.random.default_rng(seed)
    banks = []
    for h in range(H):
        l, u = float(edges[h] / 1.3), float(edges[h + 1] * 1.3)     # overlapping band for bank h
        banks.append(torch.as_tensor(l * (u / l) ** rng.random((d, K)), dtype=dtype, device=device))
    return banks


class SpectralMixture(nn.Module):
    """H-bank spectral-mixture kernel over a learned spectral embedding.

    Parameters mirror the single-bank map but are per-bank where it matters:
    frequencies ``omegas[h]`` (d, K), amplitudes ``logas[h]`` (d, K) and bandwidth
    ``log_T[h]``; the ARD relevance ``log_s`` and the encoder are shared. ``mix=False``
    makes the encoder block-diagonal (per-feature, no cross-feature mixing).
    """

    def __init__(self, d, H, K, d_phi, omegas, dtype, kernel="laplace", mix=True, d_block=None):
        super().__init__()
        self.d, self.H, self.K, self.kernel, self.mix = d, H, K, kernel, mix
        self.omegas = nn.ParameterList([nn.Parameter(o.clone()) for o in omegas])          # per-bank frequencies
        self.logas = nn.ParameterList([nn.Parameter(torch.zeros(d, K, dtype=dtype)) for _ in range(H)])
        self.log_s = nn.Parameter(torch.zeros(d, dtype=dtype))                              # shared ARD relevance
        if mix:
            self.enc = nn.Linear(d * 2 * K, d_phi).to(dtype)                                # shared full encoder (mixes features)
        else:
            self.d_block = d_block if d_block is not None else max(1, d_phi // d)
            scale = 1.0 / np.sqrt(2 * K)                                                    # shared block-diagonal encoder
            self.Wb = nn.Parameter(torch.randn(d, 2 * K, self.d_block, dtype=dtype) * scale)
            self.bb = nn.Parameter(torch.zeros(d, self.d_block, dtype=dtype))
        self.log_T = nn.Parameter(torch.zeros(H, dtype=dtype))                              # per-bank bandwidth (data-init in fit)
        self.log_w = nn.Parameter(torch.zeros(H, dtype=dtype))                              # convex fusion logits
        self.log_sig2 = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.1))), dtype=dtype))   # noise sigma^2 = ridge

    def _feats(self, x, h):
        s = F.softplus(self.log_s)
        a = F.softplus(self.logas[h])
        arg = 2 * np.pi * (s * x).unsqueeze(-1) * self.omegas[h].unsqueeze(0)               # (B, d, K)
        return torch.cat([torch.cos(arg), torch.sin(arg)], -1) * torch.cat([a, a], -1).unsqueeze(0)

    def phi_h(self, x, h):
        """Embedding of x under bank h."""
        f = self._feats(x, h)                                                              # (B, d, 2K)
        if self.mix:
            return self.enc(f.reshape(x.shape[0], -1))
        return (torch.einsum("bdk,dko->bdo", f, self.Wb) + self.bb).reshape(x.shape[0], -1)

    def embed(self, X):
        """Per-bank embeddings of X -> list of H tensors (B, d_phi)."""
        return [self.phi_h(X, h) for h in range(self.H)]

    def T(self):
        return F.softplus(self.log_T) + 1e-4

    def w(self):
        return torch.softmax(self.log_w, 0)

    def sig2(self):
        return F.softplus(self.log_sig2) + 1e-6

    def kmat(self, A_embeds, B_embeds):
        """Convex-fused kernel sum_h w_h k_h from per-bank embeddings -- a sum of H
        kernels at per-bank bandwidths (direct-sum RKHS, multi-scale)."""
        T, w = self.T(), self.w()
        sq = self.kernel == "gauss"
        out = 0.0
        for h in range(self.H):
            dist = torch.cdist(A_embeds[h], B_embeds[h])
            kh = torch.exp(-(dist * dist if sq else dist) / T[h])
            out = out + w[h] * kh
        return out

    def ard(self):
        return F.softplus(self.log_s).detach().cpu().numpy()

    def spectrum(self):
        """Per-bank learned spectral density -> (H, d, K)."""
        return np.stack([F.softplus(la).detach().cpu().numpy() for la in self.logas])
