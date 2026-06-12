"""Spectral feature-map primitives shared across the ladder.

The spectral front-end learns the spectral density of the kernel rather than
approximate a predetermined one (so it is NOT random Fourier features once the
frequencies are learned -- we never freeze random draws). For each feature j the
map carries an ARD relevance ``s_j``, a frequency grid ``omega_{j,k}`` (K per
feature) and amplitudes ``a_{j,k}`` (the spectral density), giving

    psi(x)_{j,k} = a_{j,k} * [cos, sin](2 pi s_j x_j omega_{j,k}).

The bank module that consumes this lives in :mod:`skm.mixture`. This module holds
the shared frequency-grid initializer.
"""
from __future__ import annotations

import torch


def init_omega(d, K, omega_range, seed, device, dtype):
    """Log-uniform frequency grid (d, K) spanning ``omega_range`` -- the starting
    support of the spectral density before training moves it."""
    lo, hi = omega_range
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(d, K, generator=gen).numpy()
    return torch.as_tensor(lo * (hi / lo) ** u, dtype=dtype, device=device)
