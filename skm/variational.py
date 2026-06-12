"""Variational MS-SKM: a sparse variational GP head for predictive distributions.

The NLML / KRR path gives a point prediction (the GP posterior mean). This head
trains the same learned spectral-mixture kernel by the **evidence lower bound**
(ELBO) of a sparse variational Gaussian process (Titsias inducing points, Hensman's
explicit variational posterior), so it returns a full **predictive distribution**
and scales by minibatching.

With M inducing inputs Z, a whitened variational posterior q(u) = N(L m, L S L^T)
where L = chol(K_ZZ), the variational marginal at a point is q(f_i) = N(mu_i, s_i^2),

    mu_i = b_i^T m,   s_i^2 = k_ii - b_i^T (I - S) b_i,   b_i = L^{-1} k(Z, x_i),

and the kernel's unit diagonal gives k_ii = 1. The ELBO is

    ELBO = sum_i E_{q(f_i)}[log p(y_i | f_i)] - KL(q(u) || p(u)),
    KL = 1/2 [ tr S + ||m||^2 - M - log|S| ].

Two likelihoods:
  * **regression** -- Gaussian with learned noise sigma^2; the expectation is closed
    form, predictive y* ~ N(mu*, s*^2 + sigma^2);
  * **classification** (binary) -- Bernoulli through a logistic link; the expectation
    E_{q(f)}[log Bernoulli(y|sigmoid(f))] is evaluated by Gauss-Hermite quadrature, and
    the predictive class probability is E_{q(f*)}[sigmoid(f*)] by the same quadrature.

All spectral parameters, inducing inputs and variational parameters are learned
jointly by maximizing the ELBO.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.polynomial.hermite import hermgauss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .mixture import SpectralMixture, init_banks


class VariationalMSSKM:
    """Sparse variational GP on the learned spectral-mixture kernel (ELBO-trained).

    Parameters
    ----------
    task : {"auto", "regression", "classification"}
        Classification is binary (Bernoulli likelihood, logistic link).
    H, K, d_phi, omega_range, kernel, mix : the spectral kernel (see MSSKM).
    n_inducing : int
        Number of inducing inputs M.
    n_quad : int
        Gauss-Hermite nodes for the Bernoulli expectation (classification).
    epochs, patience, batch, lr, ard_lr, freq_lr : SGD budget / learning rates.
    """

    def __init__(self, *, task="auto", H=4, K=16, d_phi=128, omega_range=(0.005, 1.0),
                 kernel="laplace", mix=True, d_block=None, n_inducing=256, n_quad=20,
                 epochs=300, patience=30, batch=1024,
                 lr=1e-2, ard_lr=5e-2, freq_lr=1e-2, jitter=1e-5,
                 seed=0, device=None, verbose=True):
        self.task_arg = task
        self.H, self.K, self.d_phi, self.omega_range = H, K, d_phi, omega_range
        self.kernel, self.mix, self.d_block = kernel, mix, d_block
        self.M, self.n_quad, self.jitter = n_inducing, n_quad, jitter
        self.epochs, self.patience, self.batch = epochs, patience, batch
        self.lr, self.ard_lr, self.freq_lr = lr, ard_lr, freq_lr
        self.seed, self.verbose = seed, verbose
        self.dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float64

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _t(self, a):
        return torch.as_tensor(a, dtype=self.dt, device=self.dev)

    # ------------------------------------------------------------ variational pieces
    def _LS(self):
        tril = torch.tril(self.LS_raw_, -1)
        diag = torch.diag_embed(F.softplus(torch.diagonal(self.LS_raw_)) + 1e-6)
        return tril + diag

    def _chol_KMM(self):
        embZ = self.smix_.embed(self.Z_)
        KMM = self.smix_.kmat(embZ, embZ)
        I = torch.eye(self.M, dtype=self.dt, device=self.dev)
        jit = self.jitter
        for _ in range(6):
            try:
                return torch.linalg.cholesky(KMM + jit * I), embZ
            except Exception:
                jit *= 10.0
        return torch.linalg.cholesky(KMM + jit * I), embZ

    def _qf(self, Xb, L, embZ):
        """Variational marginal q(f) at Xb -> (mu, s2)."""
        KMN = self.smix_.kmat(embZ, self.smix_.embed(Xb))            # (M, B)
        B = torch.linalg.solve_triangular(L, KMN, upper=False)       # b_i = L^{-1} k(Z, x_i)
        mu = B.t() @ self.m_v_
        Sv = self._LS() @ self._LS().t()
        bnorm2 = (B * B).sum(0)
        bSvb = (B * (Sv @ B)).sum(0)
        s2 = 1.0 - bnorm2 + bSvb                                     # k_ii = 1
        return mu, s2.clamp_min(1e-9)

    def _kl(self):
        LS = self._LS()
        return 0.5 * ((LS * LS).sum() + (self.m_v_ * self.m_v_).sum()
                      - self.M - 2.0 * torch.log(torch.diagonal(LS)).sum())

    def _expected_ll(self, mu, s2, yb):
        """E_{q(f)}[log p(y|f)] per point, summed over the batch."""
        if self.is_clf_:                                            # Bernoulli via Gauss-Hermite
            s = torch.sqrt(s2)
            f = mu.unsqueeze(1) + np.sqrt(2.0) * s.unsqueeze(1) * self.gh_z_   # (B, Q)
            logp1 = F.logsigmoid(f); logp0 = F.logsigmoid(-f)
            y = yb.unsqueeze(1)
            integrand = y * logp1 + (1 - y) * logp0                 # (B, Q)
            return (self.gh_w_ * integrand).sum(1).sum() / np.sqrt(np.pi)
        sig2 = F.softplus(self.log_sig2_) + 1e-6                     # Gaussian, closed form
        return (-0.5 * np.log(2 * np.pi) - 0.5 * torch.log(sig2)
                - 0.5 * ((yb - mu) ** 2 + s2) / sig2).sum()

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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler_ = StandardScaler().fit(X)
        Xtr = self._t(self.scaler_.transform(X))
        Xva = self._t(self.scaler_.transform(X_val))
        d, n = X.shape[1], len(Xtr)

        if self.is_clf_:
            self.classes_ = np.unique(np.concatenate([y, y_val]))
            if len(self.classes_) != 2:
                raise ValueError("VariationalMSSKM classification is binary only")
            cmap = {c: i for i, c in enumerate(self.classes_)}
            ytr = self._t(np.array([cmap[v] for v in y], dtype=np.float64))
            self._yva = np.array([cmap[v] for v in y_val])
            self.y_mean_, self.y_std_ = 0.0, 1.0
            z, w = hermgauss(self.n_quad)
            self.gh_z_ = self._t(z); self.gh_w_ = self._t(w)
        else:
            y = y.astype(np.float64)
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
            ytr = self._t((y - self.y_mean_) / self.y_std_)
        M = self.M = min(self.M, n)

        omegas = init_banks(d, self.K, self.H, self.omega_range, self.seed, self.dev, self.dt)
        self.smix_ = SpectralMixture(d, self.H, self.K, self.d_phi, omegas, self.dt,
                                     kernel=self.kernel, mix=self.mix, d_block=self.d_block).to(self.dev)
        with torch.no_grad():
            p0 = self.smix_.phi_h(Xtr[:min(2000, n)], 0)
            dd = torch.cdist(p0, p0)
            base = float((dd * dd).median() if self.kernel == "gauss" else dd.median())
            self.smix_.log_T.data.fill_(float(np.log(np.expm1(max(base, 1e-2)))))

        idx = np.random.default_rng(self.seed).choice(n, M, replace=False)
        self.Z_ = nn.Parameter(Xtr[idx].clone())
        self.m_v_ = nn.Parameter(torch.zeros(M, dtype=self.dt, device=self.dev))
        self.LS_raw_ = nn.Parameter(torch.eye(M, dtype=self.dt, device=self.dev)
                                    * float(np.log(np.expm1(1.0))))
        self.log_sig2_ = nn.Parameter(self._t(float(np.log(np.expm1(0.1)))))

        named = list(self.smix_.named_parameters())
        rest_p = [p for nm, p in named if nm != "log_s" and not nm.startswith("omegas")]
        var_p = [self.Z_, self.m_v_, self.LS_raw_]
        if not self.is_clf_:
            var_p.append(self.log_sig2_)
        groups = [
            {"params": [self.smix_.log_s], "lr": self.ard_lr, "weight_decay": 0.0},
            {"params": list(self.smix_.omegas), "lr": self.freq_lr, "weight_decay": 0.0},
            {"params": rest_p, "weight_decay": 0.0},
            {"params": var_p, "weight_decay": 0.0},
        ]
        opt = torch.optim.Adam(groups, lr=self.lr)

        bs = min(self.batch, n)
        best, bad, best_state = np.inf, 0, None
        ep = 0
        for ep in range(self.epochs):
            self.smix_.train()
            for bi in torch.randperm(n, device=self.dev).split(bs):
                L, embZ = self._chol_KMM()
                mu, s2 = self._qf(Xtr[bi], L, embZ)
                elbo = (n / len(bi)) * self._expected_ll(mu, s2, ytr[bi]) - self._kl()
                loss = -elbo / n
                opt.zero_grad()
                loss.backward()
                opt.step()
            vr = self._val_metric(Xva, y_val)
            if vr < best - 1e-6:
                best, bad = vr, 0
                best_state = {k: v.detach().clone() for k, v in self._state().items()}
            else:
                bad += 1
            if bad > self.patience:
                break
        self._load(best_state)
        self.smix_.eval()
        tag = "clf(Bernoulli)" if self.is_clf_ else "reg(Gauss)"
        self._log(f"[Variational MS-SKM/SVGP] {tag} H={self.H} K={self.K} M={self.M} "
                  f"val={best:.4f} ({ep+1} ep)")
        return self

    # ------------------------------------------------------------ latent / predict
    def _latent(self, X):
        """Latent predictive q(f*) -> (mu, std) over a (possibly large) input set."""
        Xt = self._t(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        with torch.no_grad():
            L, embZ = self._chol_KMM()
            mus, s2s = [], []
            for bi in torch.arange(len(Xt), device=self.dev).split(4096):
                mu, s2 = self._qf(Xt[bi], L, embZ)
                mus.append(mu); s2s.append(s2)
        return torch.cat(mus), torch.cat(s2s)

    def predict_latent(self, X):
        """Mean and standard deviation of the latent function f (epistemic uncertainty)."""
        mu, s2 = self._latent(X)
        return mu.cpu().numpy(), torch.sqrt(s2).cpu().numpy()

    # --- regression predictive ---
    def predict_dist(self, X):
        """Regression predictive (mean, variance) in the target's original units."""
        if self.is_clf_:
            raise ValueError("predict_dist is for regression; use predict_proba / predict_proba_dist")
        mu, s2 = self._latent(X)
        sig2 = float(F.softplus(self.log_sig2_) + 1e-6)
        var = (s2.cpu().numpy() + sig2) * (self.y_std_ ** 2)
        return mu.cpu().numpy() * self.y_std_ + self.y_mean_, var

    # --- classification predictive ---
    def _prob_quad(self, mu, s2):
        """E_{q(f)}[sigmoid(f)] and its predictive std, by Gauss-Hermite quadrature."""
        s = torch.sqrt(s2)
        f = mu.unsqueeze(1) + np.sqrt(2.0) * s.unsqueeze(1) * self.gh_z_
        sig = torch.sigmoid(f)
        c = 1.0 / np.sqrt(np.pi)
        p = c * (self.gh_w_ * sig).sum(1)
        p2 = c * (self.gh_w_ * sig * sig).sum(1)
        return p.clamp(1e-6, 1 - 1e-6), torch.sqrt((p2 - p * p).clamp_min(0.0))

    def predict_proba(self, X):
        if not self.is_clf_:
            raise ValueError("predict_proba is for classification")
        mu, s2 = self._latent(X)
        p1, _ = self._prob_quad(mu, s2)
        p1 = p1.cpu().numpy()
        return np.stack([1 - p1, p1], 1)

    def predict_proba_dist(self, X):
        """Mean and std of the predicted class-1 probability (uncertainty in the probability)."""
        mu, s2 = self._latent(X)
        p1, p1_std = self._prob_quad(mu, s2)
        return p1.cpu().numpy(), p1_std.cpu().numpy()

    def predict(self, X):
        if self.is_clf_:
            return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]
        mean, _ = self.predict_dist(X)
        return mean

    # ------------------------------------------------------------ metrics
    def _val_metric(self, Xva, y_val):
        if self.is_clf_:                                            # log loss: a proper scoring
            mu, s2 = self._latent_t(Xva)                            # rule, so early stopping tracks
            p1, _ = self._prob_quad(mu, s2)                         # calibration, not just the
            p1 = p1.cpu().numpy()                                   # 0.5-threshold error rate
            yv = self._yva
            return float(-np.mean(yv * np.log(p1) + (1 - yv) * np.log(1 - p1)))
        mu, _ = self._latent_t(Xva)
        p = mu.cpu().numpy() * self.y_std_ + self.y_mean_
        return float(np.sqrt(np.mean((p - y_val) ** 2)))

    def _latent_t(self, Xt):
        with torch.no_grad():
            L, embZ = self._chol_KMM()
            mu, s2 = self._qf(Xt, L, embZ)
        return mu, s2

    def score(self, X, y):
        y = np.asarray(y)
        if self.is_clf_:
            return float((self.predict(X) == y).mean())
        p = self.predict(X)
        return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

    def coverage(self, X, y, level=0.95):
        """Regression: empirical coverage of the central `level` predictive interval."""
        from scipy.stats import norm
        mean, var = self.predict_dist(X)
        z = norm.ppf(0.5 + level / 2)
        return float(np.mean(np.abs(np.asarray(y) - mean) <= z * np.sqrt(var)))

    @property
    def ard_(self):
        return self.smix_.ard()

    def _state(self):
        d = dict(self.smix_.state_dict())
        d["__Z"] = self.Z_; d["__mv"] = self.m_v_; d["__LS"] = self.LS_raw_; d["__s2"] = self.log_sig2_
        return d

    def _load(self, st):
        self.Z_.data = st["__Z"].data.clone()
        self.m_v_.data = st["__mv"].data.clone()
        self.LS_raw_.data = st["__LS"].data.clone()
        self.log_sig2_.data = st["__s2"].data.clone()
        self.smix_.load_state_dict({k: v for k, v in st.items() if not k.startswith("__")})
