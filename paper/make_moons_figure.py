"""Generate the make_moons decision-boundary figure and metrics for the paper.

    python paper/make_moons_figure.py
writes paper/figures/moons_boundaries.pdf and prints the metrics table.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skm import SpectralGAM, LearnedGAM, MSSKM, VariationalMSSKM
from catboost import CatBoostClassifier

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEED = 0
X, y = make_moons(n_samples=600, noise=0.20, random_state=SEED)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

models = {
    "rung 0: SpectralGAM\n(fixed RFF, additive)":   SpectralGAM(task="classification", seed=SEED, verbose=False),
    "rung A: LearnedGAM\n(learned freq, additive)": LearnedGAM(task="classification", seed=SEED, verbose=False),
    "rung B: MSSKM(mix=False)\n(kernel, no mixing)": MSSKM(task="classification", mix=False, seed=SEED, verbose=False),
    "rung 1: MSSKM()\n(kernel + mixing)":           MSSKM(task="classification", seed=SEED, verbose=False),
    "rung 2: MSSKM(H=4)\n(multi-bank fuse)":        MSSKM(task="classification", H=4, seed=SEED, verbose=False),
    "CatBoost\n(gradient-boosted trees)":           CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                                                                       random_seed=SEED, verbose=False),
}
for m in models.values():
    m.fit(Xtr, ytr)


def scores(m):
    proba = m.predict_proba(Xte); p1 = proba[:, 1]; pred = m.predict(Xte)
    return dict(AUC=roc_auc_score(yte, p1), F1=f1_score(yte, pred),
               Brier=brier_score_loss(yte, p1), LogLoss=log_loss(yte, proba))


x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
print(f"{'model':<26} {'AUC':>7} {'F1':>7} {'Brier':>7} {'LogLoss':>8}")
for ax, (name, m) in zip(axes.ravel(), models.items()):
    p = m.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=20, cmap="RdBu", alpha=0.75, vmin=0, vmax=1)
    ax.contour(xx, yy, p, levels=[0.5], colors="k", linewidths=1.3)
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu", edgecolor="k", s=14, linewidth=0.3)
    s = scores(m)
    ax.set_title(f"{name}\nAUC={s['AUC']:.3f}  F1={s['F1']:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{name.splitlines()[0]:<26} {s['AUC']:7.4f} {s['F1']:7.4f} {s['Brier']:7.4f} {s['LogLoss']:8.4f}")

plt.tight_layout()
out = os.path.join(FIGDIR, "moons_boundaries.pdf")
fig.savefig(out, bbox_inches="tight")
print("\nwrote", out)

# ---- predictive distribution: variational (ELBO) head, label-regression on make_moons ----
vm = VariationalMSSKM(H=2, K=16, n_inducing=120, epochs=300, patience=30, seed=SEED, verbose=False)
vm.fit(Xtr, ytr.astype(float))
mean, var = vm.predict_dist(grid)
std = np.sqrt(var).reshape(xx.shape)
mean = mean.reshape(xx.shape)

fig2, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
c1 = a1.contourf(xx, yy, np.clip(mean, 0, 1), levels=20, cmap="RdBu", alpha=0.85, vmin=0, vmax=1)
a1.contour(xx, yy, mean, levels=[0.5], colors="k", linewidths=1.3)
a1.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdBu", edgecolor="k", s=12, linewidth=0.3)
a1.set_title("predictive mean (decision boundary)", fontsize=10)
fig2.colorbar(c1, ax=a1, fraction=0.046)
c2 = a2.contourf(xx, yy, std, levels=20, cmap="magma")
a2.scatter(Xtr[:, 0], Xtr[:, 1], c="cyan", edgecolor="k", s=10, linewidth=0.2)
a2.set_title("predictive std (uncertainty)", fontsize=10)
fig2.colorbar(c2, ax=a2, fraction=0.046)
for a in (a1, a2):
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
out2 = os.path.join(FIGDIR, "moons_uncertainty.pdf")
fig2.savefig(out2, bbox_inches="tight")
print("wrote", out2)
