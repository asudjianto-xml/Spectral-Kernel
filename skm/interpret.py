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

* **Importance** --- the mean-square sensitivity of the embedding to feature
  ``j``. Differentiating an embedding coordinate ``a\\cos(2\\pi s_j\\omega x_j)``
  in ``x_j`` gives ``2\\pi s_j\\omega a\\,(\\cdot)``, so the bank-weighted
  mean-square gradient is

      I_j  \\propto  s_j^2 \\sum_h w_h \\sum_k a_{h,j,k}^2 \\omega_{h,j,k}^2 ,

  a global, derivative-based (Sobol-flavored) importance that combines the ARD
  scale with the spectral energy the model actually placed on the feature.
  Features are standardized inside the model, so the ``I_j`` are directly
  comparable across features.

For ``mix=True`` the shared encoder ``W`` mixes features, and the off-diagonal
blocks of the learned metric ``M = W^T W`` are an inspectable diagnostic of
**metric interaction** between features (Prop. ``interaction`` in the paper).
``interaction_matrix`` reports the per-feature-pair block norms. With
``mix=False`` the encoder is block-diagonal and the off-diagonal is structurally
zero.
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
        """ARD feature importance ``I_j`` (mean-square embedding sensitivity).

        Returns an array of length ``d``; normalized to sum to 1 by default.
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


def feature_importance(model, feature_names=None, normalize=True):
    """Convenience: ARD feature importance for a fitted MSSKM / VariationalMSSKM.

    Returns ``(names, importances)``.
    """
    interp = SpectralInterpreter(model, feature_names=feature_names)
    return interp.feature_names, interp.feature_importance(normalize=normalize)
