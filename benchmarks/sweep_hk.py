"""Hyperparameter sweep over H (banks) and K (frequencies per feature).

Capacity in the spectral mixture should come from H banks (multi-scale, convex-fused),
not from a large K (more frequencies in one bank overfits -- see README rung 2). This
sweep makes that trade-off visible and, crucially, selects (H, K) honestly.

Protocol (matches the tuned-CatBoost baseline):
  1. hold out a 20% TEST split (random_state=0; stratified for classification);
  2. carve a 20% VALIDATION fold from the remaining train;
  3. fit each (H, K) on train-minus-val, using val for MSSKM's internal early stopping;
  4. SELECT (H, K) by the VALIDATION metric -- never by test;
  5. refit the selected config on the full train fold and score the TEST set ONCE.
The per-cell test grid is reported as a landscape, but the headline number is the
val-selected, test-evaluated one -- so it is comparable to a val-tuned baseline.
Metric: R^2 (regression) or accuracy (classification).

    python benchmarks/sweep_hk.py                 # California, breast cancer, bike-hourly
"""
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skm import MSSKM

HS = (1, 2, 4, 8)
KS = (8, 16, 32)
SEED = 0
BIKE = os.path.expanduser("~/jupyterlab/multi-kernel/mnca/bike_hourly.npz")


def _row(label, vals):
    print(f"{label:>7} | " + " | ".join(f"{v:6.4f}" for v in vals), flush=True)


def sweep(name, X, y, task):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    clf = task == "classification"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                          stratify=(y if clf else None))
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED,
                                             stratify=(ytr if clf else None))
    metric = "acc" if clf else "R2"
    print(f"\n=== {name}  (n={len(X)}, d={X.shape[1]}, {metric}) ===", flush=True)

    val, test = {}, {}
    for H in HS:
        for K in KS:
            m = MSSKM(task=task, H=H, K=K, seed=SEED, verbose=False).fit(Xtr2, ytr2, X_val=Xva, y_val=yva)
            val[(H, K)] = m.score(Xva, yva)
            test[(H, K)] = m.score(Xte, yte)

    print(f"VAL {metric} (selection):  {'H \\ K':>5} | " + " | ".join(f"K={k:>3}" for k in KS), flush=True)
    for H in HS:
        _row(H, [val[(H, k)] for k in KS])
    print(f"TEST {metric} (landscape): {'H \\ K':>5} | " + " | ".join(f"K={k:>3}" for k in KS), flush=True)
    for H in HS:
        _row(H, [test[(H, k)] for k in KS])

    Hb, Kb = max(val, key=lambda hk: val[hk])                       # select on VALIDATION
    t0 = time.time()
    final = MSSKM(task=task, H=Hb, K=Kb, seed=SEED, verbose=False).fit(Xtr, ytr)   # refit on full train
    ft = final.score(Xte, yte)
    print(f"  val-selected: H={Hb} K={Kb}  ->  TEST {metric}={ft:.4f}   "
          f"(grid-test cell {test[(Hb, Kb)]:.4f}; refit {time.time()-t0:.0f}s)", flush=True)
    return {"val": val, "test": test, "selected": (Hb, Kb), "test_metric": ft}


def main():
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    Xc, yc = fetch_california_housing(return_X_y=True)
    sweep("California (regression)", Xc, yc, "regression")
    Xb, yb = load_breast_cancer(return_X_y=True)
    sweep("Breast cancer (classification)", Xb, yb, "classification")
    if os.path.exists(BIKE):
        d = np.load(BIKE)
        sweep("Bike-hourly (regression)", d["X"], d["y"], "regression")
    else:
        print(f"\n(bike-hourly not found at {BIKE}; skipped)")


if __name__ == "__main__":
    main()
