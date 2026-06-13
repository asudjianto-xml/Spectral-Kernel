"""ARD interpretability for the spectral kernel machines.

The spectral feature map carries a per-feature **ARD relevance** ``s_j`` (the
``log_s`` parameter), a learned spectral density ``a_{h,j,k}`` and frequencies
``omega_{h,j,k}`` per bank, and convex bank weights ``w_h``. Together these say,
without any post-hoc surrogate, how much each input feature drives the kernel.

Two readings, both from the learned kernel directly:

* **Relevance** ``s_j`` --- the ARD inverse length scale. A feature the model
  drives toward ``s_j -> 0`` varies infinitely slowly, i.e. is switched off; a
  large ``s_j`` means the embedding turns quickly with that feature. This is the
  literal ARD reading.

* **Importance** --- the gradient energy of the embedding in feature ``j``.
  Differentiating an embedding coordinate ``a*cos(2*pi*s_j*omega*x_j)`` in
  ``x_j`` gives ``2*pi*s_j*omega*a*(...)``, whose squared norm is *exactly*
  ``(2*pi*s_j*omega*a)^2`` -- a cosine/sine pair shares one constant gradient
  norm (``sin^2+cos^2=1``), independent of ``x``, so nothing is averaged or
  sampled. Summed over frequencies and the convex bank weights,

      I_j  =  (2*pi)^2 s_j^2 sum_h w_h sum_k a_{h,j,k}^2 omega_{h,j,k}^2 ,

  which is exactly the second moment of feature ``j``'s learned spectral measure
  (the GP prior's mean-square gradient). It is a closed-form functional of the
  fitted parameters -- not a surrogate, perturbation or permutation estimate.
  Features are standardized inside the model, so the ``I_j`` are directly
  comparable; they are returned normalized to sum to 1.

For ``mix=True`` the shared encoder ``W`` mixes features, and the off-diagonal
blocks of the learned metric ``M = W^T W`` are an inspectable diagnostic of
**metric interaction** between features (Prop. ``interaction`` in the paper).
``interaction_matrix`` reports the per-feature-pair block norms. With
``mix=False`` the encoder is block-diagonal and the off-diagonal is structurally
zero.

**Functional ANOVA.** The two readings above are global functionals of the learned
kernel. To see the *shape* of what the model learned, ``fanova`` decomposes the
fitted predictor itself into a grand mean, per-feature main effects and pure
pairwise interactions,

    f(x) = f_0 + sum_j f_j(x_j) + sum_{j<k} f_{jk}(x_j, x_k) + (higher order),

under the **empirical joint measure** of a reference sample (Hooker's generalized
functional ANOVA). The components are fit to the model's own predictions by penalized
backfitting over per-feature bins and the interactions are purified to be orthogonal to
their main effects, so the fit stays on the data manifold (no off-manifold extrapolation)
and the variance attribution is well behaved under feature correlation:

    total_var = sum_j Var(f_j) + sum_{j<k} Var(f_{jk}) + covariance + remainder,

with ``covariance`` the between-component dependence (zero under independence) and
``remainder`` the order-3-and-higher part --- both reported, not hidden. ``fanova``
returns a ``FanovaResult`` carrying the components and this attribution.

``main_effect`` and ``interaction_effect`` give the simpler *interventional*
(partial-dependence) curves for quick shape inspection; ``plot_main_effect`` and
``plot_interaction`` draw them. These extrapolate off-manifold under correlation, so for
variance attribution prefer ``fanova``.
"""
from __future__ import annotations

import numpy as np


def _feature_names(model, d, feature_names):
    if feature_names is not None:
        names = list(feature_names)
        if len(names) != d:
            raise ValueError(f"feature_names has {len(names)} entries, expected {d}")
        return names
    seen = getattr(model, "feature_names_in_", None)
    if seen is None:
        seen = getattr(getattr(model, "scaler_", None), "feature_names_in_", None)
    if seen is not None and len(seen) == d:
        return [str(c) for c in seen]
    return [f"x{j}" for j in range(d)]


class SpectralInterpreter:
    """Read feature relevance, importance and interactions off a fitted kernel machine.

    Parameters
    ----------
    model : MSSKM or VariationalMSSKM
        A fitted model exposing the spectral front-end ``smix_``.
    feature_names : sequence of str, optional
        Names for the ``d`` inputs. Falls back to the column names seen at fit
        time (if the model was fit on a DataFrame) and then to ``x0..x{d-1}``.
    """

    def __init__(self, model, feature_names=None):
        smix = getattr(model, "smix_", None)
        if smix is None:
            raise TypeError(
                f"{type(model).__name__} has no ARD front-end. ARD interpretation is "
                "available for MSSKM and VariationalMSSKM; the additive GAMs expose "
                "shape_function(j) instead."
            )
        self.model = model
        self.smix = smix
        self.d = int(smix.d)
        self.feature_names = _feature_names(model, self.d, feature_names)

        # everything below is read straight from the learned kernel
        self.relevance_ = smix.ard()                                   # (d,)  s_j
        self._a = smix.spectrum()                                      # (H, d, K)  amplitudes
        self._omega = np.stack([o.detach().cpu().numpy() for o in smix.omegas])  # (H, d, K)
        self._w = smix.w().detach().cpu().numpy()                      # (H,)  convex bank weights

        # per-feature spectral energy (bank-weighted), the additive marginal variance
        self.energy_ = np.einsum("h,hdk->d", self._w, self._a ** 2)    # (d,)

    # ------------------------------------------------------------ importance
    def feature_importance(self, normalize=True):
        """ARD feature importance ``I_j``, the exact gradient energy of the embedding.

        ``I_j = (2*pi)^2 s_j^2 sum_h w_h sum_k a^2 omega^2`` -- the second moment of
        feature ``j``'s learned spectral measure, computed in closed form from the
        fitted parameters (no surrogate or sampling). Length ``d``; normalized to
        sum to 1 by default.
        """
        sens = self.relevance_ ** 2 * np.einsum(
            "h,hdk,hdk->d", self._w, self._a ** 2, self._omega ** 2
        )
        if normalize:
            tot = sens.sum()
            sens = sens / tot if tot > 0 else sens
        return sens

    def ranking(self, normalize=True):
        """Features sorted by importance, as a list of ``(name, importance)`` pairs."""
        imp = self.feature_importance(normalize=normalize)
        order = np.argsort(imp)[::-1]
        return [(self.feature_names[j], float(imp[j])) for j in order]

    # ------------------------------------------------------------ interaction
    def interaction_matrix(self, normalize=True, zero_diagonal=False):
        """Per-feature-pair metric interaction from ``M = W^T W``.

        Entry ``(j, j')`` is the Frobenius norm of the ``2K x 2K`` block of the
        learned metric coupling features ``j`` and ``j'``. With ``mix=False`` the
        encoder is block-diagonal, so the off-diagonal is (structurally) zero.
        """
        d, K = self.d, int(self.smix.K)
        if getattr(self.smix, "mix", True):
            W = self.smix.enc.weight.detach().cpu().numpy()            # (d_phi, d*2K)
            M = W.T @ W                                                # (d*2K, d*2K)
            blk = M.reshape(d, 2 * K, d, 2 * K)
            inter = np.sqrt((blk ** 2).sum(axis=(1, 3)))               # (d, d) block Frobenius norm
        else:
            Wb = self.smix.Wb.detach().cpu().numpy()                   # (d, 2K, d_block)
            self_norm = np.sqrt((np.einsum("dki,dli->dkl", Wb, Wb) ** 2).sum(axis=(1, 2)))
            inter = np.diag(self_norm)                                 # block-diagonal: no cross terms
        if zero_diagonal:
            inter = inter.copy()
            np.fill_diagonal(inter, 0.0)
        if normalize:
            mx = inter.max()
            inter = inter / mx if mx > 0 else inter
        return inter

    # ------------------------------------------------------------ exact predictor decompositions
    def bank_decomposition(self, X):
        """Exact per-bank contribution to the prediction (Prop. fusion, $\\hat g_h=w_h K_h\\alpha$).

        The fused KRR predictor is $\\hat f=\\sum_h w_h k_h(\\cdot,X)\\alpha$, an exact sum
        over the $H$ banks. This returns each term, computed in closed form from the stored
        KRR coefficients --- no approximation.

        Regression: returns ``(components, intercept)`` with ``components`` of shape ``(H, N)``
        in target units and a scalar ``intercept``; ``components.sum(0) + intercept`` equals
        ``model.predict(X)`` exactly.
        Classification: returns ``(components, None)`` with ``components`` of shape ``(H, N, C)``
        of per-class score contributions; ``components.sum(0)`` is the (pre-temperature) score.
        """
        import torch
        model = self.model
        if not (hasattr(model, "coef_") and hasattr(model, "emb_tr_")):
            raise TypeError("bank_decomposition needs the KRR decode of MSSKM; "
                            "VariationalMSSKM uses inducing points, not stored coefficients.")
        Xt = model._t(model.scaler_.transform(np.asarray(X, dtype=np.float64)))
        sq = self.smix.kernel == "gauss"
        with torch.no_grad():
            emb_q = self.smix.embed(Xt)
            T, w = self.smix.T(), self.smix.w()
            comps = []
            for h in range(self.smix.H):
                dist = torch.cdist(emb_q[h], model.emb_tr_[h])
                kh = torch.exp(-(dist * dist if sq else dist) / T[h])
                comps.append((w[h] * (kh @ model.coef_)).cpu().numpy())   # (N, C)
        comps = np.stack(comps)                                            # (H, N, C)
        if model.is_clf_:
            return comps, None
        comps = comps[:, :, 0] * model.y_std_                              # (H, N) target units
        return comps, float(model.y_mean_)

    # ------------------------------------------------------------ functional ANOVA
    #
    # The MS-SKM kernel does not factor over coordinates, so the functional-ANOVA
    # marginalizations have no closed form. We evaluate them by averaging the model's
    # own predictions over a reference sample --- the interventional (partial-dependence)
    # estimator under the empirical joint measure of ``X_ref``:
    #
    #     f_0      = E[f(x)]                               (grand mean)
    #     f_j(v)   = E_{x_{-j}}[f(v, x_{-j})]   - f_0      (main effect)
    #     f_{jk}   = E_{x_{-jk}}[f(.)] - f_j - f_k - f_0   (pure pairwise interaction)
    #
    # Each expectation is the exact average over the reference rows --- no surrogate,
    # no model sampling. The components are exactly mutually orthogonal (so their
    # variances add up to Var[f]) when the features are independent. Under feature
    # correlation they are the interventional components and a covariance remainder
    # carries what they do not, which ``fanova`` reports rather than hides.

    def _resolve(self, feature):
        return feature if isinstance(feature, (int, np.integer)) else self.feature_names.index(feature)

    def _check_ref(self, X_ref):
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.ndim != 2 or X_ref.shape[1] != self.d:
            raise ValueError(f"X_ref must be (R, {self.d}); got {X_ref.shape}")
        return X_ref

    def _response(self, Z):
        """Scalar response per row: prediction in target units (regression) or
        ``P(class 1)`` (classification)."""
        Z = np.asarray(Z, dtype=np.float64)
        if getattr(self.model, "is_clf_", False):
            return self.model.predict_proba(Z)[:, 1]
        return self.model.predict(Z)

    def _response_batched(self, Z, batch):
        """Memory-bounded ``_response`` --- the kernel matrix is at most ``batch x n_train``."""
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty(len(Z))
        for s in range(0, len(Z), batch):
            out[s:s + batch] = self._response(Z[s:s + batch])
        return out

    def _pd1d(self, j, grid, X_ref, batch):
        """Partial dependence of feature ``j`` on ``grid``: ``E_{x_{-j}}[f(g, x_{-j})]``."""
        R, G = len(X_ref), len(grid)
        Z = np.tile(X_ref, (G, 1))                       # [X_ref; X_ref; ...] G blocks of R rows
        Z[:, j] = np.repeat(grid, R)
        return self._response_batched(Z, batch).reshape(G, R).mean(1)

    def _pd2d(self, j, k, gj, gk, X_ref, batch):
        """Joint partial dependence of features ``(j, k)`` on the mesh ``gj x gk``."""
        R, Gj, Gk = len(X_ref), len(gj), len(gk)
        Z = np.tile(X_ref, (Gj * Gk, 1))                 # Gj*Gk blocks of R rows, a slow / b fast
        Z[:, j] = np.repeat(np.repeat(gj, Gk), R)
        Z[:, k] = np.repeat(np.tile(gk, Gj), R)
        return self._response_batched(Z, batch).reshape(Gj, Gk, R).mean(2)

    @staticmethod
    def _grid(col, n):
        return np.linspace(np.quantile(col, 0.02), np.quantile(col, 0.98), n)

    @staticmethod
    def _bilinear(gx, gy, Z, px, py):
        """Bilinear interpolation of surface ``Z`` (len(gx) x len(gy)) at points ``(px, py)``."""
        ix = np.clip(np.searchsorted(gx, px) - 1, 0, len(gx) - 2)
        iy = np.clip(np.searchsorted(gy, py) - 1, 0, len(gy) - 2)
        x0, x1, y0, y1 = gx[ix], gx[ix + 1], gy[iy], gy[iy + 1]
        tx = np.clip((px - x0) / (x1 - x0), 0, 1)
        ty = np.clip((py - y0) / (y1 - y0), 0, 1)
        return (Z[ix, iy] * (1 - tx) * (1 - ty) + Z[ix + 1, iy] * tx * (1 - ty)
                + Z[ix, iy + 1] * (1 - tx) * ty + Z[ix + 1, iy + 1] * tx * ty)

    def main_effect(self, feature, X_ref, grid=None, n_grid=50, center=True, batch=4096):
        """Interventional partial-dependence curve of one feature.

        ``f_j(v) = E_{x_{-j}}[f(v, x_{-j})] - f_0``, with the expectation an exact average
        over the rows of ``X_ref``, the feature swept across ``grid`` while the others keep
        their observed values. This is the marginal (interventional) effect; it queries the
        model off the data manifold under feature correlation, so for a dependence-aware,
        variance-attributing decomposition use :meth:`fanova` instead. When ``center`` is set
        the curve is shifted to integrate to zero under the marginal of feature ``j``.

        Parameters
        ----------
        feature : int or str
            Feature index or name.
        X_ref : array (R, d)
            Reference rows in the original (unstandardized) feature units.
        grid : array, optional
            Sweep values; defaults to ``n_grid`` points spanning the 2--98% range in ``X_ref``.
        center : bool
            Center the effect under the marginal of feature ``j``.
        batch : int
            Prediction batch size (bounds the kernel matrix to ``batch x n_train``).

        Returns ``(grid, effect)``; for classification the effect is on ``P(class 1)``.
        """
        j = self._resolve(feature)
        X_ref = self._check_ref(X_ref)
        if grid is None:
            grid = self._grid(X_ref[:, j], n_grid)
        else:
            grid = np.asarray(grid, dtype=np.float64)
        effect = self._pd1d(j, grid, X_ref, batch)
        if center:
            effect = effect - np.interp(X_ref[:, j], grid, effect).mean()
        return grid, effect

    def interaction_effect(self, feat_j, feat_k, X_ref, grids=None, n_grid=20,
                           center=True, batch=4096):
        """Interventional pairwise interaction surface of features ``(j, k)``.

        ``f_{jk}(a, b) = E_{x_{-jk}}[f(.)] - f_j(a) - f_k(b) - f_0`` --- the joint partial
        dependence with the two main effects and the grand mean removed, so what remains is
        the part of the response that neither feature explains alone. Evaluated as
        interventional averages over ``X_ref`` (off-manifold under correlation); for the
        dependence-aware decomposition use :meth:`fanova`.

        Returns ``(grid_j, grid_k, surface)`` where ``surface`` has shape
        ``(len(grid_j), len(grid_k))``.
        """
        j, k = self._resolve(feat_j), self._resolve(feat_k)
        X_ref = self._check_ref(X_ref)
        if grids is None:
            gj, gk = self._grid(X_ref[:, j], n_grid), self._grid(X_ref[:, k], n_grid)
        else:
            gj, gk = (np.asarray(g, dtype=np.float64) for g in grids)
        f0 = self._response_batched(X_ref, batch).mean()
        _, fj = self.main_effect(j, X_ref, grid=gj, center=True, batch=batch)
        _, fk = self.main_effect(k, X_ref, grid=gk, center=True, batch=batch)
        surface = self._pd2d(j, k, gj, gk, X_ref, batch) - f0
        surface = surface - fj[:, None] - fk[None, :]
        if center:
            at_data = self._bilinear(gj, gk, surface, X_ref[:, j], X_ref[:, k])
            surface = surface - at_data.mean()
        return gj, gk, surface

    def fanova(self, X_ref, features=None, interactions=True, max_interaction_features=6,
               n_bins=16, ref_size=4000, batch=4096, seed=0, smooth_main=1.0, smooth_pair=4.0,
               backfit_iters=25, purify_iters=40, tol=1e-10):
        """Generalized functional-ANOVA decomposition with variance attribution.

        Decomposes the fitted predictor under the **empirical joint measure** of ``X_ref``
        into a grand mean, per-feature main effects and pure pairwise interactions,

            f(x) = f_0 + sum_j g_j(x_j) + sum_{j<k} g_{jk}(x_j, x_k) + residual.

        The components are fit to the model's own predictions at the reference rows by
        **penalized backfitting** over per-feature quantile bins: each main effect is a
        roughness-penalized 1-D smoother of the partial residual, each interaction a
        roughness-penalized 2-D smoother. The penalty keeps the effective degrees of freedom
        low, so the 2-D estimates do not overfit the way free per-cell means do. Because the fit
        only ever uses feature combinations that actually occur, it stays on-manifold (no
        extrapolation). Each interaction is then *purified* (Hooker's generalized fANOVA /
        Lengerich et al.): its conditional means are removed until
        ``E[g_{jk} | x_j] = E[g_{jk} | x_k] = 0`` and the backfit re-absorbs them into the
        mains, so every interaction is orthogonal to its own main effects and the constant under
        the empirical measure.

        Variance attribution (all under the empirical measure):

            total_var = sum_j Var(g_j) + sum_{j<k} Var(g_{jk}) + covariance + remainder.

        ``covariance`` is the cross-covariance between components --- zero when the features are
        independent, otherwise the genuine dependence signal. ``remainder`` is what an additive
        + pairwise model on these bins cannot reach (order-3 and higher interactions, plus
        within-bin variation). Both are reported rather than folded away.

        Parameters
        ----------
        X_ref : array (R, d)
            Reference rows in original units; subsampled to ``ref_size``.
        features : sequence of int/str, optional
            Features to take main effects of (default: all).
        interactions : bool
            Whether to compute pairwise interactions.
        max_interaction_features : int
            Cap on the number of features (the most ARD-relevant) entered into pairwise
            interactions; all ``C(m, 2)`` pairs among them are computed.
        n_bins : int
            Number of equal-mass (quantile) bins per feature.
        ref_size : int
            Reference rows used (subsampled deterministically by ``seed``).
        batch : int
            Prediction batch size.
        smooth_main, smooth_pair : float
            Roughness-penalty strengths for the 1-D and 2-D smoothers, relative to the bin
            counts (larger = smoother, lower effective degrees of freedom).
        backfit_iters, purify_iters, tol : int, int, float
            Backfitting and purification iteration caps and convergence tolerance.

        Returns a :class:`FanovaResult`.
        """
        X = self._check_ref(X_ref)
        if len(X) > ref_size:
            idx = np.random.RandomState(seed).choice(len(X), ref_size, replace=False)
            X = X[np.sort(idx)]
        feats = list(range(self.d)) if features is None else [self._resolve(f) for f in features]
        y = self._response_batched(X, batch)
        N = len(y)
        f0 = float(y.mean())
        total_var = float(y.var())

        # ---- equal-mass quantile bins; bin index, mass and center per feature
        binidx, mass, center = {}, {}, {}
        all_feats = set(feats)
        if interactions:
            imp = self.feature_importance(normalize=False)
            pool = sorted(feats, key=lambda j: imp[j], reverse=True)[:max_interaction_features]
            pairs = [(pool[a], pool[b]) for a in range(len(pool)) for b in range(a + 1, len(pool))]
            all_feats |= {j for pr in pairs for j in pr}
        else:
            pairs = []
        for j in all_feats:
            col = X[:, j]
            e = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
            if len(e) < 2:
                e = np.array([col.min(), col.max() + 1e-9])
            bi = np.clip(np.digitize(col, e[1:-1]), 0, len(e) - 2)
            nb = len(e) - 1
            cnt = np.bincount(bi, minlength=nb).astype(float)
            ctr = np.array([col[bi == a].mean() if cnt[a] > 0 else 0.5 * (e[a] + e[a + 1])
                            for a in range(nb)])
            binidx[j], mass[j], center[j] = bi, cnt / N, ctr

        # ---- penalized second-difference roughness operator per feature
        def _pen(nb):
            if nb < 3:
                return np.zeros((nb, nb))
            D = np.zeros((nb - 2, nb))
            for r in range(nb - 2):
                D[r, r], D[r, r + 1], D[r, r + 2] = 1.0, -2.0, 1.0
            return D.T @ D

        cnt1 = {j: np.bincount(binidx[j], minlength=len(center[j])).astype(float) for j in all_feats}
        pen1 = {j: _pen(len(center[j])) for j in all_feats}
        cnt2, A2 = {}, {}
        for (j, k) in pairs:
            nbj, nbk = len(center[j]), len(center[k])
            flat = binidx[j] * nbk + binidx[k]
            cnt2[(j, k)] = np.bincount(flat, minlength=nbj * nbk).reshape(nbj, nbk).astype(float)
            Pj, Pk = pen1[j], pen1[k]
            A2[(j, k)] = (np.kron(Pj, np.eye(nbk)) + np.kron(np.eye(nbj), Pk))  # 2D roughness

        eps = total_var * 1e-9 + 1e-12

        def smooth1d(j, resid):
            nb = len(center[j])
            b = np.bincount(binidx[j], weights=resid, minlength=nb)
            lam = smooth_main * cnt1[j].mean()                       # scale penalty to bin counts
            s = np.linalg.solve(np.diag(cnt1[j]) + lam * pen1[j] + eps * np.eye(nb), b)
            return s - float((mass[j] * s).sum())                    # weighted-center

        def smooth2d(j, k, resid):
            nbj, nbk = len(center[j]), len(center[k])
            flat = binidx[j] * nbk + binidx[k]
            b = np.bincount(flat, weights=resid, minlength=nbj * nbk)
            W = cnt2[(j, k)].reshape(-1)
            lam = smooth_pair * (N / (nbj * nbk))                    # scale penalty to mean cell count
            t = np.linalg.solve(np.diag(W) + lam * A2[(j, k)] + eps * np.eye(nbj * nbk), b)
            T = t.reshape(nbj, nbk)
            w = cnt2[(j, k)] / N                                     # purify: zero conditional means;
            wj, wk = w.sum(1), w.sum(0)                              # the backfit re-absorbs them into mains
            for _ in range(purify_iters):
                rm = np.where(wj > 0, (w * T).sum(1) / np.maximum(wj, 1e-300), 0.0)
                T = T - rm[:, None]
                cm = np.where(wk > 0, (w * T).sum(0) / np.maximum(wk, 1e-300), 0.0)
                T = T - cm[None, :]
                if max(np.abs(rm).max(initial=0.0), np.abs(cm).max(initial=0.0)) < tol:
                    break
            return T

        # ---- backfitting: penalized main + pairwise fit to the predictions
        g = {j: np.zeros(len(center[j])) for j in all_feats}
        gp = {p: np.zeros((len(center[p[0]]), len(center[p[1]]))) for p in pairs}
        fit_m = {j: np.zeros(N) for j in all_feats}
        fit_p = {p: np.zeros(N) for p in pairs}
        recon = np.full(N, f0)
        for _ in range(backfit_iters):
            delta = 0.0
            for j in all_feats:
                pr = y - recon + fit_m[j]
                g[j] = smooth1d(j, pr)
                new = g[j][binidx[j]]
                recon = recon - fit_m[j] + new; delta = max(delta, np.abs(new - fit_m[j]).max())
                fit_m[j] = new
            for (j, k) in pairs:
                pr = y - recon + fit_p[(j, k)]
                gp[(j, k)] = smooth2d(j, k, pr)                      # purified; removed means flow to residual
                new = gp[(j, k)][binidx[j], binidx[k]]
                recon = recon - fit_p[(j, k)] + new
                delta = max(delta, np.abs(new - fit_p[(j, k)]).max(initial=0.0))
                fit_p[(j, k)] = new
            if delta < tol:
                break

        # ---- re-center mains to zero empirical mean (absorb into f_0)
        for j in all_feats:
            mu = float((mass[j] * g[j]).sum())
            g[j] = g[j] - mu
            f0 += mu
        recon = np.full(N, f0)
        for j in all_feats:
            recon = recon + g[j][binidx[j]]
        for p in pairs:
            recon = recon + gp[p][binidx[p[0]], binidx[p[1]]]

        # ---- variance attribution under the empirical measure
        mains = {j: (center[j], g[j], float((mass[j] * g[j] ** 2).sum())) for j in feats}
        inters = {(j, k): (center[j], center[k], gp[(j, k)],
                           float(((cnt2[(j, k)] / N) * gp[(j, k)] ** 2).sum())) for (j, k) in pairs}
        explained = float(np.var(recon))
        comp_var = sum(v[2] for v in mains.values()) + sum(v[3] for v in inters.values())
        covariance = explained - comp_var
        remainder = total_var - explained

        return FanovaResult(self.feature_names, f0, total_var, mains, inters,
                            covariance, remainder)

    # ------------------------------------------------------------ reporting
    def summary(self, top=None):
        """A text table of features ranked by ARD importance (with relevance and energy)."""
        imp = self.feature_importance(normalize=True)
        order = np.argsort(imp)[::-1]
        if top is not None:
            order = order[:top]
        width = max(7, max(len(self.feature_names[j]) for j in order))
        lines = [f"{'feature':>{width}}  {'importance':>10}  {'relevance':>9}  {'energy':>8}",
                 f"{'-' * width}  {'-' * 10}  {'-' * 9}  {'-' * 8}"]
        for j in order:
            lines.append(f"{self.feature_names[j]:>{width}}  {imp[j]:>10.4f}  "
                         f"{self.relevance_[j]:>9.4f}  {self.energy_[j]:>8.4f}")
        return "\n".join(lines)

    def __repr__(self):
        return f"SpectralInterpreter(d={self.d}, H={self.smix.H}, K={self.smix.K})"

    # ------------------------------------------------------------ plots
    def plot_importance(self, ax=None, top=None, color="#3b7ea1"):
        """Horizontal bar chart of ARD feature importance."""
        import matplotlib.pyplot as plt
        imp = self.feature_importance(normalize=True)
        order = np.argsort(imp)[::-1]
        if top is not None:
            order = order[:top]
        order = order[::-1]                                            # largest on top
        if ax is None:
            _, ax = plt.subplots(figsize=(6, max(2.0, 0.32 * len(order))))
        ax.barh([self.feature_names[j] for j in order], imp[order], color=color)
        ax.set_xlabel("ARD importance (normalized)")
        ax.set_title("Feature importance")
        return ax

    def plot_interactions(self, ax=None, zero_diagonal=True, cmap="magma"):
        """Heatmap of the metric interaction matrix."""
        import matplotlib.pyplot as plt
        M = self.interaction_matrix(normalize=True, zero_diagonal=zero_diagonal)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(M, cmap=cmap)
        ax.set_xticks(range(self.d)); ax.set_yticks(range(self.d))
        ax.set_xticklabels(self.feature_names, rotation=90, fontsize=8)
        ax.set_yticklabels(self.feature_names, fontsize=8)
        ax.set_title("Metric interaction $\\|M_{jj'}\\|_F$")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        return ax

    def plot_main_effect(self, feature, X_ref, ax=None, n_grid=50, color="#3b7ea1", **kw):
        """Line plot of one feature's interventional partial-dependence curve."""
        import matplotlib.pyplot as plt
        j = self._resolve(feature)
        grid, effect = self.main_effect(j, X_ref, n_grid=n_grid, **kw)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(grid, effect, color=color, lw=2)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xlabel(self.feature_names[j])
        ylab = "effect on $P(y{=}1)$" if getattr(self.model, "is_clf_", False) else "effect on $\\hat y$"
        ax.set_ylabel(ylab)
        ax.set_title(f"Main effect: {self.feature_names[j]}")
        return ax

    def plot_interaction(self, feat_j, feat_k, X_ref, ax=None, n_grid=20,
                         cmap="RdBu_r", **kw):
        """Filled-contour plot of an interventional pairwise interaction surface."""
        import matplotlib.pyplot as plt
        j, k = self._resolve(feat_j), self._resolve(feat_k)
        gj, gk, surf = self.interaction_effect(j, k, X_ref, n_grid=n_grid, **kw)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 4.2))
        lim = float(np.abs(surf).max()) or 1.0
        im = ax.contourf(gj, gk, surf.T, levels=14, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xlabel(self.feature_names[j])
        ax.set_ylabel(self.feature_names[k])
        ax.set_title(f"Interaction: {self.feature_names[j]} $\\times$ {self.feature_names[k]}")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        return ax


class FanovaResult:
    """Result of :meth:`SpectralInterpreter.fanova` --- the generalized functional-ANOVA
    components (under the empirical joint measure) and their variance attribution.

    Attributes
    ----------
    f0 : float
        Grand mean ``E[f]`` over the reference sample.
    total_var : float
        Variance of the prediction over the reference sample.
    explained_var : float
        Variance of the reconstructed additive + pairwise model.
    covariance : float
        Cross-covariance between components --- zero under feature independence, otherwise the
        dependence signal. Equals ``explained_var - sum of component variances``.
    remainder : float
        ``total_var - explained_var`` --- order-3-and-higher interactions and within-bin
        variation that an additive + pairwise model on these bins cannot reach.
    main : dict
        ``name -> (centers, values, variance)`` for each main effect (``values`` over bin
        ``centers``).
    interaction : dict
        ``(name_j, name_k) -> (centers_j, centers_k, surface, variance)`` for each pairwise term.

    The attribution balances exactly:
    ``total_var = sum Var(main) + sum Var(pair) + covariance + remainder``.
    """

    def __init__(self, names, f0, total_var, mains, inters, covariance, remainder):
        self._names = names
        self.f0 = f0
        self.total_var = total_var
        self.covariance = covariance
        self.remainder = remainder
        self.explained_var = total_var - remainder
        self.main = {names[j]: v for j, v in mains.items()}
        self.interaction = {(names[j], names[k]): v for (j, k), v in inters.items()}

    def importance(self, normalize=True):
        """Components ranked by attributed variance, as ``(label, variance)`` pairs.

        With ``normalize`` the variances are divided by ``total_var`` so they read as the
        fraction of prediction variance each component carries.
        """
        items = [(n, v[2]) for n, v in self.main.items()]
        items += [(f"{a} x {b}", v[3]) for (a, b), v in self.interaction.items()]
        denom = self.total_var if (normalize and self.total_var > 0) else 1.0
        items = [(n, val / denom) for n, val in items]
        return sorted(items, key=lambda t: t[1], reverse=True)

    def summary(self, top=None):
        """A text table of components ranked by their share of prediction variance."""
        rows = self.importance(normalize=True)
        if top is not None:
            rows = rows[:top]
        denom = self.total_var or 1.0
        width = max(11, max((len(n) for n, _ in rows), default=11))
        lines = [f"{'component':>{width}}  {'var share':>9}",
                 f"{'-' * width}  {'-' * 9}"]
        for n, frac in rows:
            lines.append(f"{n:>{width}}  {frac:>9.4f}")
        lines.append(f"{'-' * width}  {'-' * 9}")
        lines.append(f"{'correlation':>{width}}  {self.covariance / denom:>9.4f}")
        lines.append(f"{'remainder':>{width}}  {self.remainder / denom:>9.4f}")
        lines.append(f"{'f0':>{width}}  {self.f0:>9.4f}   total_var={self.total_var:.4g}")
        return "\n".join(lines)

    def plot_main(self, feature, ax=None, color="#3b7ea1"):
        """Step plot of a purified main effect over its bin centers."""
        import matplotlib.pyplot as plt
        centers, values, _ = self.main[feature]
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(centers, values, color=color, lw=2, marker="o", ms=3)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xlabel(feature)
        ax.set_ylabel("effect")
        ax.set_title(f"Main effect: {feature}")
        return ax

    def plot_interaction(self, feat_j, feat_k, ax=None, cmap="RdBu_r"):
        """Heatmap of a purified pairwise interaction surface."""
        import matplotlib.pyplot as plt
        key = (feat_j, feat_k) if (feat_j, feat_k) in self.interaction else (feat_k, feat_j)
        cj, ck, surf, _ = self.interaction[key]
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 4.2))
        lim = float(np.abs(surf).max()) or 1.0
        im = ax.pcolormesh(cj, ck, surf.T, cmap=cmap, vmin=-lim, vmax=lim, shading="auto")
        ax.set_xlabel(key[0]); ax.set_ylabel(key[1])
        ax.set_title(f"Interaction: {key[0]} $\\times$ {key[1]}")
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        return ax

    def __repr__(self):
        return (f"FanovaResult(f0={self.f0:.4g}, total_var={self.total_var:.4g}, "
                f"explained={self.explained_var / (self.total_var or 1.0):.1%}, "
                f"{len(self.main)} main, {len(self.interaction)} pairwise)")


def feature_importance(model, feature_names=None, normalize=True):
    """Convenience: ARD feature importance for a fitted MSSKM / VariationalMSSKM.

    Returns ``(names, importances)``.
    """
    interp = SpectralInterpreter(model, feature_names=feature_names)
    return interp.feature_names, interp.feature_importance(normalize=normalize)


def fanova(model, X_ref, feature_names=None, **kw):
    """Convenience: functional-ANOVA decomposition of a fitted MSSKM.

    Returns a :class:`FanovaResult`. See :meth:`SpectralInterpreter.fanova`.
    """
    return SpectralInterpreter(model, feature_names=feature_names).fanova(X_ref, **kw)
