"""Generate the ARD interpretability figure for the paper.

    python paper/make_interpret_figure.py
writes paper/figures/interpret.pdf (ARD feature importance + metric interaction)
and prints the ranked summary, fitting MSSKM on California Housing.
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
print(f"test R2 = {m.score(Xte, yte):.4f}\n")
print(itp.summary())

imp = itp.feature_importance()
order = np.argsort(imp)                                          # ascending; largest on top of barh
names = itp.feature_names
M = itp.interaction_matrix(zero_diagonal=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.barh([names[j] for j in order], imp[order], color="#3b7ea1")
a1.set_xlabel("ARD importance (normalized)")
a1.set_title("Feature importance", fontsize=11)

im = a2.imshow(M, cmap="magma")
a2.set_xticks(range(len(names))); a2.set_yticks(range(len(names)))
a2.set_xticklabels(names, rotation=90, fontsize=8)
a2.set_yticklabels(names, fontsize=8)
a2.set_title(r"Metric interaction $\|M_{jj'}\|_F$ (off-diagonal)", fontsize=11)
fig.colorbar(im, ax=a2, fraction=0.046)

plt.tight_layout()
out = os.path.join(FIGDIR, "interpret.pdf")
fig.savefig(out, bbox_inches="tight")
print("\nwrote", out)
