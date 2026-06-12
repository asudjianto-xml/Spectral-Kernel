"""Multi-Scale Spectral Kernel Machine (MS-SKM).

A fully-learned kernel machine for tabular data. The spectral density of the kernel
(per-feature relevance, frequencies, amplitudes) is learned end to end; the kernel
is a distance kernel on the learned embedding; the model is trained by the GP
marginal likelihood (NLML) and decoded by full-train kernel ridge regression at the
learned noise (= ridge).

    phi_h(x) = SpectralMixture bank h                              # learned spectral density
    K        = sum_h w_h exp(-||phi_h(x) - phi_h(x')|| / T_h)      # convex-fused spectral mixture
    train    : minimize NLML over the banks, T_h, w_h, sigma^2     # GP marginal likelihood
    predict  : K(x, train) (K_train + sigma^2 I)^{-1} Y            # kernel ridge

``H=1`` is the single-bank kernel machine (rung 1); ``H>1`` fuses banks at different
scales into one spectral-mixture kernel (rung 2). ``mix=False`` holds the encoder
block-diagonal so the embedding carries no cross-feature mixing (rung B).

Leakage guards: the feature/label scalers are fit on train only; the ridge and the
calibration temperature are selected on a validation fold; test data is touched only
at predict.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .mixture import SpectralMixture, init_banks
from .linalg import krr_solve


class MSSKM:
    """Spectral kernel machine with H frequency banks.

    Parameters
    ----------
    task : {"auto", "regression", "classification"}
    H : int
        Number of frequency banks. 1 = single bank; >1 = fused spectral mixture.
    K : int
        Frequencies per feature per bank. Small by design -- large K overfits;
        16 per variable is a good default, capacity comes from H banks not from K.
    d_phi : int
        Encoder embedding dimension.
    omega_range : (float, float)
        Log-uniform range for the initial frequency grid; banks split it.
    kernel : {"laplace", "gauss"}
        Per-bank stationary kernel on the embedding distance.
    mix : bool
        Full encoder mixing features (True) or block-diagonal per-feature encoder (False).
    d_block : int or None
        Per-feature embedding width when ``mix=False``; defaults to ``d_phi // d``.
    learn_freq : bool
        Reserved; frequencies are always learned in the mixture front-end.
    cg_rank : int
        Lanczos rank for the KRR decode.
    epochs, patience, m_nlml : training budget. m_nlml is the NLML support-subset size.
    lr, ard_lr, freq_lr : learning rates for the encoder, the ARD scale, the frequencies.
    """

    def __init__(self, *, task="auto", H=1, K=16, d_phi=128, omega_range=(0.005, 1.0),
                 kernel="laplace", mix=True, d_block=None, learn_freq=True, cg_rank=150,
                 epochs=200, patience=20, m_nlml=2048,
                 lr=1e-2, ard_lr=5e-2, freq_lr=1e-2,
                 calibrate=True, seed=0, device=None, verbose=True):
        self.task_arg = task
        self.H, self.K, self.d_phi, self.omega_range = H, K, d_phi, omega_range
        self.kernel, self.mix, self.d_block = kernel, mix, d_block
        self.learn_freq, self.cg_rank = learn_freq, cg_rank
        self.epochs, self.patience, self.m_nlml = epochs, patience, m_nlml
        self.lr, self.ard_lr, self.freq_lr = lr, ard_lr, freq_lr
        self.calibrate, self.seed, self.verbose = calibrate, seed, verbose
        self.dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float64                       # NLML Cholesky needs double precision

    # ---------------------------------------------------------------- helpers
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _t(self, a, dtype=None):
        return torch.as_tensor(a, dtype=dtype or self.dt, device=self.dev)

    def _denorm(self, p):
        return p * self.y_std_ + self.y_mean_

    def _metric(self, scores, y_raw):
        if self.is_clf_:
            return 1.0 - float((scores.argmax(1).cpu().numpy() == self._yva_idx).mean())
        p = self._denorm(scores[:, 0].detach().cpu().numpy())
        return float(np.sqrt(np.mean((p - np.asarray(y_raw, dtype=np.float64)) ** 2)))

    def _chol(self, A, I):
        """Cholesky with jitter retry for PD safety."""
        jit = 1e-6
        for _ in range(6):
            try:
                return torch.linalg.cholesky(A + jit * I)
            except Exception:
                jit *= 10.0
        return torch.linalg.cholesky(A + jit * I)

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
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
            self.y_mean_, self.y_std_ = 0.0, 1.0
            Y = self._t(np.eye(len(self.classes_))[[cmap[v] for v in y]])
        else:
            y = y.astype(np.float64)
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
            Y = self._t((y - self.y_mean_) / self.y_std_)[:, None]
        C = Y.shape[1]

        omegas = init_banks(d, self.K, self.H, self.omega_range, self.seed, self.dev, self.dt)
        self.smix_ = SpectralMixture(d, self.H, self.K, self.d_phi, omegas, self.dt,
                                     kernel=self.kernel, mix=self.mix, d_block=self.d_block).to(self.dev)

        # data-init every bank's bandwidth from bank 0's embedding distance scale
        with torch.no_grad():
            p0 = self.smix_.phi_h(Xtr[:min(2000, n)], 0)
            dd = torch.cdist(p0, p0)
            base = float((dd * dd).median() if self.kernel == "gauss" else dd.median())
            self.smix_.log_T.data.fill_(float(np.log(np.expm1(max(base, 1e-2)))))

        # scale/relevance/frequency params are weight-decay-free; ARD and frequencies
        # get their own learning rate. encoder + amplitudes get ridge weight decay.
        enc_params = list(self.smix_.enc.parameters()) if self.mix else [self.smix_.Wb, self.smix_.bb]
        groups = [
            {"params": [self.smix_.log_s], "lr": self.ard_lr, "weight_decay": 0.0},
            {"params": list(self.smix_.omegas), "lr": self.freq_lr, "weight_decay": 0.0},
            {"params": [self.smix_.log_T, self.smix_.log_w, self.smix_.log_sig2], "weight_decay": 0.0},
            {"params": enc_params + list(self.smix_.logas), "weight_decay": 2e-4},
        ]
        opt = torch.optim.AdamW(groups, lr=self.lr)

        m = min(self.m_nlml, n)
        steps_per_epoch = max(1, n // m)
        Im = torch.eye(m, dtype=self.dt, device=self.dev)
        Sval = torch.as_tensor(np.random.default_rng(self.seed).choice(n, m, replace=False), device=self.dev)
        YSval = Y[Sval]
        best, bad, best_state = np.inf, 0, None
        ep = 0
        for ep in range(self.epochs):
            self.smix_.train()
            for _ in range(steps_per_epoch):
                S = torch.randperm(n, device=self.dev)[:m]
                sig2 = self.smix_.sig2()
                emb = self.smix_.embed(Xtr[S])
                Kr = self.smix_.kmat(emb, emb) + sig2 * Im
                L = self._chol(Kr, Im)
                YS = Y[S]
                al = torch.cholesky_solve(YS, L)
                nlml = 0.5 * (YS * al).sum() + 0.5 * C * 2 * torch.log(torch.diagonal(L)).sum()
                loss = nlml / (m * C)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vr = self._val(Xtr, Xva, Sval, YSval, Im, y_val)
            if vr < best - 1e-6:
                best, bad = vr, 0
                best_state = {k: v.detach().clone() for k, v in self.smix_.state_dict().items()}
            else:
                bad += 1
            if bad > self.patience:
                break

        self.smix_.load_state_dict(best_state)
        self.smix_.eval()
        tag = "clf" if self.is_clf_ else "reg"
        wts = np.round(self.smix_.w().detach().cpu().numpy(), 3)
        self._log(f"[MS-SKM] task={tag} H={self.H} K={self.K} d_phi={self.d_phi} "
                  f"mix={self.mix} kernel={self.kernel} val={best:.4f} ({ep + 1} ep) w={wts}")

        # full-train fused KRR decode at the learned ridge
        with torch.no_grad():
            self.lam_ = float(self.smix_.sig2())
            self.emb_tr_ = self.smix_.embed(Xtr)
            Kfull = self.smix_.kmat(self.emb_tr_, self.emb_tr_)
            self.coef_ = krr_solve(Kfull, Y, self.lam_, min(self.cg_rank, n), self.dev, self.dt)

        self.T_cal_ = 1.0
        if self.is_clf_ and self.calibrate:
            self.T_cal_ = self._fit_temperature(Xva)
        return self

    def _val(self, Xtr, Xva, Sval, YSval, Im, y_val):
        """Cheap validation: KRR on the fixed m-point support subset, predict val."""
        with torch.no_grad():
            sig2 = self.smix_.sig2()
            embS = self.smix_.embed(Xtr[Sval])
            embV = self.smix_.embed(Xva)
            Kss = self.smix_.kmat(embS, embS) + sig2 * Im
            al = torch.cholesky_solve(YSval, self._chol(Kss, Im))
            sv = self.smix_.kmat(embV, embS) @ al
        return self._metric(sv, y_val)

    # ---------------------------------------------------------------- decode / predict
    def _scores(self, X):
        Xt = self._t(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        with torch.no_grad():
            emb_q = self.smix_.embed(Xt)
            Kqt = self.smix_.kmat(emb_q, self.emb_tr_)
            return Kqt @ self.coef_

    def predict(self, X):
        s = self._scores(X)
        if self.is_clf_:
            return self.classes_[s.argmax(1).cpu().numpy()]
        return self._denorm(s[:, 0].cpu().numpy())

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

    def _fit_temperature(self, Xva):
        with torch.no_grad():
            emb_v = self.smix_.embed(Xva)
            s = (self.smix_.kmat(emb_v, self.emb_tr_) @ self.coef_).cpu().numpy()
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

    # ---------------------------------------------------------------- interpretability
    @property
    def ard_(self):
        """Per-feature relevance s_j (shared across banks)."""
        return self.smix_.ard()

    @property
    def spectrum_(self):
        """Per-bank learned spectral density, shape (H, d, K)."""
        return self.smix_.spectrum()

    @property
    def bank_weights_(self):
        """Convex fusion weights w_h over the H banks."""
        return self.smix_.w().detach().cpu().numpy()
