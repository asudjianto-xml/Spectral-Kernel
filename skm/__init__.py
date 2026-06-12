"""skm -- Multi-Scale Spectral Kernel Machines for tabular data.

A fully-learned kernel machine that aims to be a competitive alternative to GBDT.
The kernel's spectral density (per-feature relevance, frequencies, amplitudes) is
learned end to end; the model is trained by the GP marginal likelihood and decoded
by kernel ridge regression.

The ladder, basic to sophisticated:
  * ``SpectralGAM`` -- rung 0: fixed random Fourier features, no mixing, no kernel;
    a closed-form ridge GAM (additive main effects).
  * ``LearnedGAM`` -- rung A: learned frequencies, still additive and kernel-free;
    an SGD-trained GAM that places its basis frequencies where each feature needs them.
  * ``MSSKM`` -- single-bank spectral kernel machine: learned spectral density,
    distance kernel on the embedding, NLML-trained, KRR-decoded.
"""
from .gam import SpectralGAM
from .learned_gam import LearnedGAM
from .model import MSSKM
from .variational import VariationalMSSKM

__all__ = ["SpectralGAM", "LearnedGAM", "MSSKM", "VariationalMSSKM"]
__version__ = "0.0.1"
