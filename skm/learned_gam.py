"""LearnedGAM -- rung A: a GAM that learns its frequencies.

One restriction lifted from rung 0: the frequencies are no longer fixed random
draws, they are learned. The model stays additive (no feature mixing) and
kernel-free (a linear readout), so it is still a GAM -- but now it places its
basis frequencies where each feature needs them instead of spreading them
randomly. The per-feature shape function

    f_j(x_j) = sum_k [ alpha_{j,k} cos(2 pi omega_{j,k} x_j)
                       + beta_{j,k} sin(2 pi omega_{j,k} x_j) ]

learns both the frequencies omega_{j,k} (the support of the spectral density) and
the amplitudes (alpha, beta) -- which, with a linear readout, ARE the readout
weights. Learning omega makes the model nonlinear in its parameters, so unlike
rung 0 this trains by SGD rather than closed form.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .features import init_omega


class _AdditiveSpectral(nn.Module):
    """Per-feature spectral map with learnable frequencies -> additive linear readout.

    The readout is a single Linear over the flattened per-feature features. Because
    feature block j depends only on x_j, the output is additive: the readout weights
    of block j are exactly the learned amplitudes of f_j. No cross-feature mixing.
    """

    def __init__(self, d, K, C, omega_init):
        super().__init__()
        self.d, self.K, self.C = d, K, C
        self.omega = nn.Parameter(omega_init.clone())     # (d, K) learned frequencies
        self.readout = nn.Linear(d * 2 * K, C)            # additive: weights = learned amplitudes

    def features(self, x):
        arg = 2 * np.pi * x.unsqueeze(-1) * self.omega.unsqueeze(0)   # (n, d, K), per-feature
        feats = torch.cat([torch.cos(arg), torch.sin(arg)], -1)       # (n, d, 2K)
        return feats.reshape(x.shape[0], -1)

    def forward(self, x):
        return self.readout(self.features(x))


class LearnedGAM:
    """Additive spectral GAM with learned frequencies (rung A).

    Parameters
    ----------
    task : {"auto", "regression", "classification"}
    K : int
        Frequencies per feature.
    omega_range : (float, float)
        Log-uniform range for the initial frequency grid (learned thereafter).
    epochs, patience, batch : SGD budget.
    lr : readout / base learning rate.
    freq_lr : learning rate for the frequencies (own group, no weight decay).
    weight_decay : ridge on the readout (the amplitudes).
    seed, device, verbose
    """

    def __init__(self, *, task="auto", K=64, omega_range=(0.005, 1.0),
                 epochs=300, patience=30, batch=1024,
                 lr=5e-3, freq_lr=5e-3, weight_decay=1e-3,
                 calibrate=True, seed=0, device=None, verbose=True):
        self.task_arg = task
        self.K, self.omega_range = K, omega_range
        self.epochs, self.patience, self.batch = epochs, patience, batch
        self.lr, self.freq_lr, self.weight_decay = lr, freq_lr, weight_decay
        self.calibrate, self.seed, self.verbose = calibrate, seed, verbose
        self.dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float32

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _t(self, a, dtype=None):
        return torch.as_tensor(a, dtype=dtype or self.dt, device=self.dev)

    # ------------------------------------------------------------ fit
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if self.task_arg == "auto":
            uniq = np.unique(y)
            self.is_clf_ = (y.dtype.kind in "iuOSUb") or (len(uniq) <= 20 and np.allclose(uniq, uniq.astype(int)))
        else:
            self.is_clf_ = self.task_arg == "classification"

        if X_val is None:
            strat = y if self.is_clf_ else None
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=strat)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler_ = StandardScaler().fit(X)
        Xtr = self._t(self.scaler_.transform(X))
        Xva = self._t(self.scaler_.transform(X_val))
        d, n = X.shape[1], len(Xtr)

        if self.is_clf_:
            self.classes_ = np.unique(np.concatenate([y, y_val]))
            cmap = {c: i for i, c in enumerate(self.classes_)}
            self._yva_idx = np.array([cmap[v] for v in y_val])
            ytr_idx = self._t([cmap[v] for v in y], torch.long)
            self.y_mean_, self.y_std_ = 0.0, 1.0
            C = len(self.classes_)
        else:
            y = y.astype(np.float32)
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
            ytr = self._t((y - self.y_mean_) / self.y_std_)
            C = 1

        omega = init_omega(d, self.K, self.omega_range, self.seed, self.dev, self.dt)
        self.net_ = _AdditiveSpectral(d, self.K, C, omega).to(self.dev)

        # frequencies: own LR, no weight decay (decaying them toward 0 is wrong).
        # readout (the amplitudes): weight decay = ridge.
        groups = [
            {"params": [self.net_.omega], "lr": self.freq_lr, "weight_decay": 0.0},
            {"params": self.net_.readout.parameters(), "weight_decay": self.weight_decay},
        ]
        opt = torch.optim.AdamW(groups, lr=self.lr)

        bs = min(self.batch, max(2, n // 2))
        best, bad, best_state = np.inf, 0, None
        ep = 0
        for ep in range(self.epochs):
            self.net_.train()
            for bi in torch.randperm(n, device=self.dev).split(bs):
                out = self.net_(Xtr[bi])
                loss = (F.cross_entropy(out, ytr_idx[bi]) if self.is_clf_
                        else F.mse_loss(out.squeeze(-1), ytr[bi]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vr = self._metric(self.net_(Xva), y_val)
            if vr < best - 1e-6:
                best, bad = vr, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
            if bad > self.patience:
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        tag = "clf" if self.is_clf_ else "reg"
        self._log(f"[LearnedGAM rungA] task={tag} K={self.K} val={best:.4f} ({ep + 1} ep)")
        self.T_cal_ = 1.0
        if self.is_clf_ and self.calibrate:
            self.T_cal_ = self._fit_temperature(Xva)
        return self

    # ------------------------------------------------------------ predict
    def _scores(self, X):
        Xt = self._t(self.scaler_.transform(np.asarray(X, dtype=np.float32)))
        with torch.no_grad():
            return self.net_(Xt)

    def predict(self, X):
        s = self._scores(X)
        if self.is_clf_:
            return self.classes_[s.argmax(1).cpu().numpy()]
        return s[:, 0].cpu().numpy() * self.y_std_ + self.y_mean_

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

        Returns ``(grid, f_j)``; f_j is the feature's additive contribution to f(x),
        per class for classification (shape (n_grid, C))."""
        if grid is None:
            lo = self.scaler_.mean_[j] - 3 * self.scaler_.scale_[j]
            hi = self.scaler_.mean_[j] + 3 * self.scaler_.scale_[j]
            grid = np.linspace(lo, hi, n_grid)
        grid = np.asarray(grid, dtype=np.float64)
        xs = (grid - self.scaler_.mean_[j]) / self.scaler_.scale_[j]
        xt = self._t(xs)
        with torch.no_grad():
            arg = 2 * np.pi * xt.unsqueeze(-1) * self.net_.omega[j].unsqueeze(0)   # (n_grid, K)
            feats = torch.cat([torch.cos(arg), torch.sin(arg)], -1)               # (n_grid, 2K)
            W = self.net_.readout.weight                                          # (C, d*2K)
            b0 = j * 2 * self.K
            Wj = W[:, b0:b0 + 2 * self.K]                                         # (C, 2K)
            fj = feats @ Wj.t()                                                   # (n_grid, C)
        if not self.is_clf_:
            fj = fj * self.y_std_
        fj = fj.cpu().numpy()
        if center:
            fj = fj - fj.mean(0, keepdims=True)
        return grid, (fj[:, 0] if not self.is_clf_ else fj)

    # ------------------------------------------------------------ internals
    def _metric(self, scores, y_raw):
        if self.is_clf_:
            return 1.0 - float((scores.argmax(1).cpu().numpy() == self._yva_idx).mean())
        p = scores[:, 0].detach().cpu().numpy() * self.y_std_ + self.y_mean_
        return float(np.sqrt(np.mean((p - np.asarray(y_raw, dtype=np.float32)) ** 2)))

    def _fit_temperature(self, Xva):
        with torch.no_grad():
            s = self.net_(Xva).cpu().numpy()
        yidx, rows = self._yva_idx, np.arange(len(self._yva_idx))

        def nll(T):
            z = s / T
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            return -np.mean(np.log(np.clip(p[rows, yidx], 1e-12, 1)))

        grid = np.logspace(-1.5, 2, 80)
        T0 = min(grid, key=nll)
        return float(min(np.concatenate([[T0], np.linspace(T0 / 1.5, T0 * 1.5, 60)]), key=nll))
