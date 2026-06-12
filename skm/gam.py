"""SpectralGAM -- rung 0: a Generalized Additive Model from random Fourier features.

The simplest member of the family. Three deliberate restrictions strip the model
down to a GAM:

  1. **Fixed frequencies (RFF style).** Per-feature frequencies are random Gaussian
     draws, fixed at init -- not learned. This is classic random Fourier features:
     phi approximates an RBF kernel of a chosen length scale.
  2. **No mixing.** Each feature gets its own 1-D Fourier expansion psi_j(x_j); no
     encoder mixes features. The model retains main effects only.
  3. **No kernel.** A direct linear readout on the features, solved in closed form
     by ridge regression -- no distance kernel, no SGD.

The result is additive,

    f(x) = b + sum_j f_j(x_j),   f_j(x_j) = sum_k [a_{j,k} cos(w_{j,k} x_j) + b_{j,k} sin(w_{j,k} x_j)],

so each per-feature shape function f_j is directly recoverable and plottable -- the
defining interpretability property of a GAM. Frequencies are fixed, so the design
matrix is fixed and the ridge readout has an exact closed form; the ridge lambda
and the length scale are selected on a validation fold.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SpectralGAM:
    """Random-Fourier-feature GAM with a closed-form ridge readout.

    Parameters
    ----------
    task : {"auto", "regression", "classification"}
    K : int
        Frequencies per feature (the expansion adds a cos and a sin per frequency,
        so each feature contributes 2K columns).
    length_scale : float or sequence of float
        RBF length scale(s) of the random frequencies, on standardized features. A
        sequence is searched on the validation fold. Smaller -> wigglier shapes.
    lambdas : sequence of float
        Ridge values searched on the validation fold (closed-form sweep).
    seed, device, verbose
    """

    def __init__(self, *, task="auto", K=64, length_scale=(0.5, 1.0, 2.0, 4.0),
                 lambdas=None, seed=0, device=None, verbose=True):
        self.task_arg = task
        self.K = K
        self.length_scales = np.atleast_1d(np.asarray(length_scale, dtype=float))
        self.lambdas = np.asarray(lambdas) if lambdas is not None else np.logspace(-4, 4, 41)
        self.seed, self.verbose = seed, verbose
        self.dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float64

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _t(self, a):
        return torch.as_tensor(a, dtype=self.dt, device=self.dev)

    # ------------------------------------------------------------ features
    def _features(self, Xs, omega):
        """Per-feature random Fourier features -> (n, d * 2K). No mixing: column
        block j depends only on x_j."""
        arg = Xs.unsqueeze(-1) * omega.unsqueeze(0)                 # (n, d, K), per-feature
        feats = torch.cat([torch.cos(arg), torch.sin(arg)], -1)     # (n, d, 2K)
        feats = feats * np.sqrt(1.0 / self.K)                       # RFF normalization
        return feats.reshape(Xs.shape[0], -1)

    def _draw_omega(self, d, length_scale):
        """Fixed Gaussian frequencies (d, K) for an RBF kernel of the given length
        scale: w ~ N(0, 1/length_scale^2)."""
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        w = torch.randn(d, self.K, generator=gen, dtype=torch.float64).numpy()
        return self._t(w / length_scale)

    def _design(self, Xs, omega):
        """Feature matrix with a leading intercept column."""
        Phi = self._features(Xs, omega)
        ones = torch.ones(Phi.shape[0], 1, dtype=self.dt, device=self.dev)
        return torch.cat([ones, Phi], 1)

    # ------------------------------------------------------------ fit
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.task_arg == "auto":
            uniq = np.unique(y)
            self.is_clf_ = (y.dtype.kind in "iuOSUb") or (len(uniq) <= 20 and np.allclose(uniq, uniq.astype(int)))
        else:
            self.is_clf_ = self.task_arg == "classification"

        if X_val is None:
            strat = y if self.is_clf_ else None
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=strat)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self._t(self.scaler_.transform(X))
        Xvs = self._t(self.scaler_.transform(X_val))
        d = X.shape[1]

        if self.is_clf_:
            self.classes_ = np.unique(np.concatenate([y, y_val]))
            cmap = {c: i for i, c in enumerate(self.classes_)}
            Y = self._t(np.eye(len(self.classes_))[[cmap[v] for v in y]])     # one-hot
            yva_idx = np.array([cmap[v] for v in y_val])
            self.y_mean_, self.y_std_ = 0.0, 1.0
        else:
            y = y.astype(np.float64)
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
            Y = self._t((y - self.y_mean_) / self.y_std_)[:, None]
        C = Y.shape[1]

        # search (length_scale, lambda): each length scale fixes the features. One
        # eigendecomposition of the P x P Gram G = Phi^T Phi then sweeps every lambda
        # in closed form -- ridge(lambda) = V diag(1/(e+lambda)) V^T (Phi^T Y).
        best = (np.inf, None, None, None)        # (val_loss, length_scale, lambda, omega)
        for ls in self.length_scales:
            omega = self._draw_omega(d, ls)
            Ptr = self._design(Xs, omega)         # (n, P)
            Pva = self._design(Xvs, omega)        # (nval, P)
            G = Ptr.t() @ Ptr                     # (P, P)
            e, V = torch.linalg.eigh(G)           # ascending eigenvalues e >= 0
            c = V.t() @ (Ptr.t() @ Y)             # (P, C)
            PvaV = Pva @ V                        # (nval, P)
            for lam in self.lambdas:
                sv = PvaV @ (c / (e + lam).unsqueeze(1))
                loss = self._metric(sv, y_val, yva_idx if self.is_clf_ else None)
                if loss < best[0]:
                    best = (loss, ls, lam, omega)

        _, self.ls_, self.lam_, self.omega_ = best

        # final readout on ALL provided training data (train + val) at the selected
        # (length_scale, lambda) -- standard refit, uses every labeled row.
        Xall = np.concatenate([X, X_val], 0)
        Yall = torch.cat([Y, self._onehot_or_norm(y_val)], 0)
        Xas = self._t(self.scaler_.transform(Xall))
        Pall = self._design(Xas, self.omega_)
        G = Pall.t() @ Pall + self.lam_ * torch.eye(Pall.shape[1], dtype=self.dt, device=self.dev)
        self.beta_ = torch.linalg.solve(G, Pall.t() @ Yall)        # (P, C)

        tag = "clf" if self.is_clf_ else "reg"
        self._log(f"[SpectralGAM rung0] task={tag} K={self.K} length_scale={self.ls_:.3g} "
                  f"lambda={self.lam_:.3g} val={best[0]:.4f}")
        self.T_cal_ = 1.0
        if self.is_clf_:
            self.T_cal_ = self._fit_temperature(Xvs, yva_idx)
        return self

    def _onehot_or_norm(self, yv):
        if self.is_clf_:
            cmap = {c: i for i, c in enumerate(self.classes_)}
            return self._t(np.eye(len(self.classes_))[[cmap[v] for v in yv]])
        return self._t((yv.astype(np.float64) - self.y_mean_) / self.y_std_)[:, None]

    # ------------------------------------------------------------ predict
    def _scores(self, X):
        Xs = self._t(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        return self._design(Xs, self.omega_) @ self.beta_

    def predict(self, X):
        s = self._scores(X)
        if self.is_clf_:
            return self.classes_[s.argmax(1).cpu().numpy()]
        return (s[:, 0].cpu().numpy() * self.y_std_ + self.y_mean_)

    def predict_proba(self, X, calibrated=True):
        z = self._scores(X).cpu().numpy()
        T = self.T_cal_ if (calibrated and getattr(self, "T_cal_", 1.0)) else 1.0
        z = z / T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def score(self, X, y):
        y = np.asarray(y)
        if self.is_clf_:
            return float((self.predict(X) == y).mean())
        p = self.predict(X)
        return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

    # ------------------------------------------------------------ GAM interpretability
    def shape_function(self, j, grid=None, n_grid=200, center=True):
        """The learned per-feature shape function f_j over a grid of x_j (raw units).

        Returns ``(grid, f_j)``. Because the model is additive, f_j is exactly the
        feature's contribution to f(x). For classification f_j is returned per class
        (shape (n_grid, C)).
        """
        if grid is None:
            lo, hi = self.scaler_.mean_[j] - 3 * self.scaler_.scale_[j], self.scaler_.mean_[j] + 3 * self.scaler_.scale_[j]
            grid = np.linspace(lo, hi, n_grid)
        grid = np.asarray(grid, dtype=np.float64)
        xs = (grid - self.scaler_.mean_[j]) / self.scaler_.scale_[j]      # standardize feature j
        xt = self._t(xs)
        arg = xt.unsqueeze(-1) * self.omega_[j].unsqueeze(0)             # (n_grid, K)
        feats = torch.cat([torch.cos(arg), torch.sin(arg)], -1) * np.sqrt(1.0 / self.K)
        # beta block for feature j: intercept(1) + d blocks of 2K each
        b0 = 1 + j * 2 * self.K
        beta_j = self.beta_[b0:b0 + 2 * self.K]                          # (2K, C)
        fj = (feats @ beta_j)                                           # (n_grid, C)
        if not self.is_clf_:
            fj = fj * self.y_std_                                       # back to target units
        fj = fj.cpu().numpy()
        if center:
            fj = fj - fj.mean(0, keepdims=True)
        return grid, (fj[:, 0] if (not self.is_clf_) else fj)

    # ------------------------------------------------------------ internals
    def _metric(self, scores, y_raw, yva_idx):
        if self.is_clf_:
            return 1.0 - float((scores.argmax(1).cpu().numpy() == yva_idx).mean())
        p = scores[:, 0].detach().cpu().numpy() * self.y_std_ + self.y_mean_
        return float(np.sqrt(np.mean((p - np.asarray(y_raw, dtype=np.float64)) ** 2)))

    def _fit_temperature(self, Xvs, yva_idx):
        s = (self._design(Xvs, self.omega_) @ self.beta_).cpu().numpy()
        rows = np.arange(len(yva_idx))

        def nll(T):
            z = s / T
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            return -np.mean(np.log(np.clip(p[rows, yva_idx], 1e-12, 1)))

        grid = np.logspace(-1.5, 2, 80)
        T0 = min(grid, key=nll)
        return float(min(np.concatenate([[T0], np.linspace(T0 / 1.5, T0 * 1.5, 60)]), key=nll))
