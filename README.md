# skm — Multi-Scale Spectral Kernel Machines

A fully-learned kernel machine for tabular data, built to be a competitive
alternative to gradient-boosted trees. The kernel's **spectral density** —
per-feature relevance, frequencies and amplitudes — is learned end to end. The
model is trained by the GP marginal likelihood and decoded by kernel ridge
regression.

This repository builds the framework from the most basic rung up.

## Install

```bash
pip install git+https://github.com/asudjianto-xml/Spectral-Kernel.git
```

Or from a clone:

```bash
git clone https://github.com/asudjianto-xml/Spectral-Kernel.git
cd Spectral-Kernel
pip install -e .                 # core (numpy, scipy, scikit-learn, torch)
pip install -e ".[bench]"        # + CatBoost / Optuna for benchmarks
pip install -e ".[tutorial]"     # + matplotlib / jupyter for the notebook
```

The package depends on PyTorch. For a GPU build, install the CUDA wheel that
matches your platform from [pytorch.org](https://pytorch.org) before (or after)
installing `skm`; the default `torch>=2.0` dependency pulls the CPU build.

```python
from skm import MSSKM

m = MSSKM(task="auto").fit(X_train, y_train)
print(m.score(X_test, y_test))
```

## The ladder

| rung | model | frequencies | mixing | kernel | readout | state |
|---|---|---|---|---|---|---|
| 0 | `SpectralGAM` | fixed (RFF) | none (additive) | none | closed-form ridge | **done** |
| A | `LearnedGAM` | learned | none (additive) | none | SGD (linear) | **done** |
| B | `MSSKM(mix=False)` | learned | block-diagonal | distance kernel on φ | NLML + KRR | **done** |
| 1 | `MSSKM` | learned | full encoder | distance kernel on φ | NLML + KRR | **done** |
| 2 | `MSSKM(H>1)` | learned, H banks | full encoder | convex-fused spectral mixture | NLML + KRR | **done** |

Each rung lifts exactly one restriction. Rung 0 is a fixed-frequency GAM; rung A
learns the frequencies but stays additive; rung B turns the **kernel** on while
keeping the embedding block-diagonal (no feature mixing), so any interactions are
axis-aligned; rung 1 lets the encoder mix features, so interactions become oblique;
rung 2 fuses banks.

A linear layer alone never creates interactions and is invisible to a linear
readout (composition of linear maps is linear). Interactions come from the
**kernel** — its `exp(-‖φ(x)-φ(x')‖)` is a nonlinearity whose product/cross terms
couple features. Mixing does not toggle interactions on or off; with the kernel it
only rotates them from axis-aligned (rung B) to oblique (rung 1). Rungs A→B→1
measure that decomposition.

## Rung 0 — SpectralGAM (RFF-GAM)

The simplest member. Three restrictions strip the model to a GAM:

1. **Fixed frequencies (RFF style)** — per-feature frequencies are random Gaussian
   draws, fixed, approximating an RBF kernel of a chosen length scale. Not learned.
2. **No mixing** — each feature gets its own 1-D Fourier expansion `psi_j(x_j)`; no
   encoder mixes features. Main effects only.
3. **No kernel** — a direct linear readout, solved in closed form by ridge.

```
f(x) = b + sum_j f_j(x_j),   f_j(x_j) = sum_k [a_{j,k} cos(w_{j,k} x_j) + b_{j,k} sin(w_{j,k} x_j)]
```

Because frequencies are fixed the design matrix is fixed, so the ridge readout is
exact (`Phi^T Phi` eigendecomposition); `lambda` and the length scale are selected
on a validation fold. Each per-feature shape function `f_j` is directly recoverable.

```python
from skm import SpectralGAM

m = SpectralGAM(task="regression").fit(X_train, y_train)
print(m.score(X_test, y_test))
grid, f0 = m.shape_function(0)        # the learned 1-D shape for feature 0, in target units
```

## Rung A — LearnedGAM (learned-frequency GAM)

One restriction lifted: the frequencies are learned instead of fixed. The model
stays additive and kernel-free, so it is still a GAM, but it places its basis
frequencies where each feature needs them. With a linear readout the readout
weights ARE the learned amplitudes, so this rung learns both frequency and
amplitude. Learning the frequencies makes it nonlinear in its parameters, so it
trains by SGD.

```python
from skm import LearnedGAM

m = LearnedGAM(task="regression").fit(X_train, y_train)
grid, f0 = m.shape_function(0)
```

On California and breast cancer, learning the frequencies matches fixed RFF
(0.711 vs 0.713, 0.947 vs 0.947): in the additive regime these tasks are bounded
by the additive structure, not the frequency placement. The lift that closes the
gap to the kernel is interactions, not learned frequencies — see rung 1.

## Rung B — MSSKM(mix=False) (kernel on, no mixing)

One restriction lifted from rung A: the **kernel** is turned on, but the encoder
is held **block-diagonal** — each feature j gets its own embedding `phi_j` from
`x_j` alone, with no cross-feature weights. The kernel `exp(-‖phi(x)-phi(x')‖/T)`
is a nonlinearity, so interactions appear (a product kernel over per-feature
embeddings contains all interaction orders), but they are axis-aligned. Comparing
rung B to rung 1 isolates exactly what the oblique mixing buys.

```python
from skm import MSSKM
m = MSSKM(mix=False).fit(X_train, y_train)   # kernel, block-diagonal embedding
```

## Rung 1 — MSSKM (single-bank spectral kernel machine)

```
phi(x)  = SpectralFeatures(x)                       # learned spectral density -> embedding
K(x,x') = exp(-||phi(x) - phi(x')|| / T)            # Laplace (or Gaussian) kernel on phi
train   : minimize NLML over phi, T, sigma^2         # exact GP marginal likelihood
predict : K(x, train) (K_train + sigma^2 I)^{-1} Y   # full-train kernel ridge
```

The spectral feature map (`skm/features.py`) is the one novel piece. For each
feature `j` it learns:

- an **ARD relevance** `s_j = softplus(.)` — how much the feature matters;
- a **frequency grid** `omega_{j,k}` (K per feature), log-spaced at init then learned —
  the support of the spectral density;
- **amplitudes** `a_{j,k} = softplus(.)` — the spectral density itself.

`psi(x)_{j,k} = a_{j,k} · [cos, sin](2π · s_j · x_j · omega_{j,k})`, followed by a
linear encoder `phi = W psi`. The encoder mixes the per-feature spectral
coordinates, so feature interactions come from the kernel acting on `phi` — no
MLP depth. Because frequencies are learned (never frozen random draws), this is
**not** random Fourier features: we learn the kernel, not approximate a fixed one.

## Usage

```python
from skm import MSSKM

m = MSSKM(task="regression").fit(X_train, y_train)   # task="auto" also works
print(m.score(X_test, y_test))                       # R2 (reg) / accuracy (clf)

m.ard_         # per-feature relevance s_j
m.spectrum_    # learned spectral density a_{j,k}, shape (d, K)
```

Classification uses one-hot KRR with temperature-scaled `predict_proba`.

## Interpretability — ARD

The kernel is learned, so its parameters *are* the explanation. `SpectralInterpreter`
reads feature importance and interactions straight off a fitted `MSSKM` or
`VariationalMSSKM` — no surrogate, no permutation.

```python
from skm import MSSKM, SpectralInterpreter

m = MSSKM(task="regression").fit(X_train, y_train)   # pass a DataFrame to keep names
itp = SpectralInterpreter(m)

itp.feature_importance()      # per-feature ARD importance (normalized, sums to 1)
itp.ranking()                 # [(name, importance), ...] sorted
print(itp.summary(top=10))    # importance + raw relevance s_j + spectral energy
itp.interaction_matrix()      # d×d metric interaction ‖M_jj'‖ from the encoder

itp.plot_importance(top=10)   # bar chart   (needs matplotlib)
itp.plot_interactions()       # heatmap
```

Two ARD readings come directly from the spectral map:

- **Relevance** `s_j` — the ARD inverse length scale. `s_j → 0` switches a feature off
  (its embedding coordinate stops turning); a large `s_j` means the embedding turns
  quickly with that feature.
- **Importance** `I_j ∝ s_j² · Σ_h w_h Σ_k a²_{h,j,k} ω²_{h,j,k}` — the bank-weighted
  mean-square sensitivity of the embedding to feature `j` (a derivative-based global
  index). It combines the ARD scale with the spectral energy the model actually placed
  on the feature; inputs are standardized internally, so the `I_j` are comparable.

For `mix=True` the off-diagonal blocks of the learned metric `M = WᵀW` measure
**metric interaction** between feature pairs (Prop. 2 in the paper); with `mix=False`
the encoder is block-diagonal and the off-diagonal is structurally zero. The additive
GAMs have no ARD scale — use their `shape_function(j)` instead.

## Status

Smoke test (`python tests/test_smoke.py`, GPU), test-set numbers:

| dataset | rung 0 GAM | rung A LearnedGAM | rung B kernel/no-mix | rung 1 MSSKM |
|---|---|---|---|---|
| California housing (R²) | 0.713 | 0.711 | 0.873 | 0.876 |
| Breast cancer (acc) | 0.947 | 0.947 | 0.947 | 0.965 |

The A→B→1 decomposition splits the gain into "kernel on" vs "oblique mixing", and
the split is opposite across datasets:

- **California**: the kernel captures ~all of it (0.711 → 0.873), mixing adds ~nothing
  (→ 0.876). The interactions are axis-aligned — the product kernel reaches them
  without mixing features.
- **Breast cancer**: the kernel with no mixing adds nothing (0.947 → 0.947); the
  whole gain comes from oblique mixing (→ 0.965). The interactions live along learned
  feature combinations, not raw axes.

This is the point of the ladder: a single number hides whether a model needs
interactions at all, and if so whether axis-aligned or oblique.

## Rung 2 — multi-bank (`MSSKM(H>1)`)

H frequency banks, each its own band `ω_h`, embedding `φ_h` and bandwidth `T_h`,
fused convexly into one spectral-mixture kernel `K = Σ_h w_h k_h`.

| config | California R² | Breast cancer acc |
|---|---|---|
| H=1 single bank | 0.876 | 0.965 |
| H=4 convex fuse | 0.885 | 0.956 |
| H=8 convex fuse | 0.885 | 0.921 |

Multi-bank helps where multi-scale structure exists (California, +0.009, weights
spread and decay low→high frequency) and hurts where it does not (breast cancer,
n=455 — weights stay near-uniform, the extra banks only add capacity that overfits).

**Why convex fuse, and not a linear mixer.** Combining the H banks with a single
linear mixer collapses them: a linear map over the concatenated bank features is one
encoder over `H·K` pooled frequencies → one embedding → **one kernel with one
bandwidth**. That is identical to concatenation, which is identical to a single bank
with `K' = H·K`. One bandwidth cannot serve frequencies spanning many scales, so the
pooled kernel overfits or saturates. The convex fuse is the only one of these that
does *not* collapse: it keeps H separate kernels, each with its own bandwidth `T_h`,
and sums them. A sum of kernels at different scales is irreducible to any single
kernel — that is exactly why multi-bank is not the same as more frequencies in one
bank.

## Layout

```
skm/
  gam.py        SpectralGAM: fixed RFF, no mixing, no kernel -> closed-form ridge GAM (rung 0)
  learned_gam.py LearnedGAM: learned frequencies, additive, no kernel -> SGD GAM (rung A)
  features.py   init_omega: shared log-uniform frequency-grid initializer
  mixture.py    SpectralMixture: H banks, shared ARD+encoder, per-bank T_h, convex fuse
  linalg.py     Lanczos tridiagonalization + KRR solve / ridge sweep
  model.py      MSSKM: H>=1 banks; fit / predict / predict_proba / score (mix=False -> rung B)
tests/
  test_smoke.py rungs 0, A, 1, 2, regression + classification sanity check
benchmarks/
  sweep_hk.py        H x K sweep (banks vs frequencies-per-feature), val-selected
  tenseed_compare.py 10-seed MS-SKM vs tuned CatBoost (R2/MSE/MAD, ACC/AUC/LogLoss/Brier)
  catboost_baseline.py  single-split Optuna-tuned CatBoost
tutorial/
  msskm_moons_tutorial.ipynb  decision boundaries on make_moons across the ladder + CatBoost
  build_tutorial.py           regenerates the notebook
paper/
  msskm.tex     self-contained paper (theory + ladder experiments); references.bib
```

## Tutorial

`tutorial/msskm_moons_tutorial.ipynb` walks the ladder on 2-D `make_moons`, where the
decision boundary of each rung is plottable: the additive rungs give blocky,
near-separable boundaries; the kernel curves the boundary to follow the moons; CatBoost
draws an axis-aligned staircase. Regenerate with `python tutorial/build_tutorial.py`
then execute with `jupyter nbconvert --to notebook --execute --inplace`.

## Tests

```bash
pip install -e ".[bench,tutorial,dev]"
python tests/test_smoke.py        # rungs 0, A, 1, 2 — regression + classification
```

Runs on CPU; a CUDA-enabled PyTorch build is used automatically when available.
