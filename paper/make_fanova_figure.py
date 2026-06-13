"""Generate the functional-ANOVA figure for the paper.

    python paper/make_fanova_figure.py
writes paper/figures/fanova.pdf (variance attribution + a main effect + top interaction)
and prints the decomposition, fitting MSSKM on California Housing (H=4, K=8).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skm import MSSKM, SpectralInterpreter

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SEED = 0

data = fetch_california_housing(as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=SEED)
m = MSSKM(task="regression", H=4, K=8, seed=SEED, verbose=False).fit(Xtr, ytr)   # val-selected config
itp = SpectralInterpreter(m)
res = itp.fanova(Xtr, ref_size=4000, n_bins=16, max_interaction_features=5)
print(f"test R2 = {m.score(Xte, yte):.4f}\n")
print(res.summary(top=12))
print(f"\nexplained = {res.explained_var / res.total_var:.3f}")

# ---- variance attribution: top components + correlation + remainder
rows = res.importance(normalize=True)[:6]
labels = [n for n, _ in rows] + ["correlation", "remainder"]
vals = [v for _, v in rows] + [res.covariance / res.total_var, res.remainder / res.total_var]
colors = ["#3b7ea1"] * len(rows) + ["#9aa0a6", "#c0c0c0"]

main_feat = rows[0][0]                                            # top main effect
pair = max(res.interaction, key=lambda k: res.interaction[k][3])  # top interaction

fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13, 3.8))

order = np.argsort(vals)
a0.barh([labels[i] for i in order], [vals[i] for i in order], color=[colors[i] for i in order])
a0.set_xlabel("share of prediction variance")
a0.set_title("Functional-ANOVA attribution", fontsize=11)

centers, values, var = res.main[main_feat]
a1.plot(centers, values, color="#3b7ea1", lw=2, marker="o", ms=3)
a1.axhline(0.0, color="0.7", lw=0.8, zorder=0)
a1.set_xlabel(main_feat); a1.set_ylabel("effect on price")
a1.set_title(f"Main effect: {main_feat}", fontsize=11)

cj, ck, surf, _ = res.interaction[pair]
lim = float(np.abs(surf).max()) or 1.0
im = a2.pcolormesh(cj, ck, surf.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
a2.set_xlabel(pair[0]); a2.set_ylabel(pair[1])
a2.set_title(f"Interaction: {pair[0]}$\\times${pair[1]}", fontsize=11)
fig.colorbar(im, ax=a2, fraction=0.046)

plt.tight_layout()
out = os.path.join(FIGDIR, "fanova.pdf")
fig.savefig(out, bbox_inches="tight")
print("\nwrote", out)
