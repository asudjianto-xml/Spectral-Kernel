"""Ten-seed comparison: MS-SKM at chosen (H, K) vs tuned CatBoost.

Ten independent train/test splits (seeds 0..9, test_size=0.2, stratified for
classification). For each split:
  * MS-SKM is fit at the chosen (H, K) for that dataset -- a single hyperparameter
    choice, selected once on a separate validation fold (see benchmarks/sweep_hk.py);
  * CatBoost is freshly tuned on that split's own train/validation fold (Optuna,
    early stopping) and refit -- a per-seed tuned baseline.
This is generous to CatBoost (re-tuned every split) and fixed for MS-SKM, so the
comparison does not favor the proposed model.

Metrics, reported as mean +/- std over the ten seeds:
  * regression:     R^2, MSE, MAD (mean absolute deviation)
  * classification: accuracy, AUC, log loss, Brier score

    python benchmarks/tenseed_compare.py --model msskm      # GPU
    python benchmarks/tenseed_compare.py --model catboost   # CPU

msskm (GPU) and catboost (CPU) can run concurrently on a dedicated node.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, roc_auc_score, log_loss, brier_score_loss)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEEDS = list(range(10))
BIKE = os.path.expanduser("~/jupyterlab/multi-kernel/mnca/bike_hourly.npz")

# chosen MS-SKM (H, K) per dataset -- validation-selected on the canonical split
# (benchmarks/sweep_hk.py). Bike updated once that sweep completes.
PARAMS = {
    "California": (4, 8),
    "Breast cancer": (2, 32),
    "Bike-hourly": (8, 16),
}

REG_METRICS = ["R2", "MSE", "MAD"]
CLF_METRICS = ["ACC", "AUC", "LogLoss", "Brier"]


def datasets():
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    out = []
    Xc, yc = fetch_california_housing(return_X_y=True)
    out.append(("California", Xc, yc, "regression"))
    Xb, yb = load_breast_cancer(return_X_y=True)
    out.append(("Breast cancer", Xb, yb, "classification"))
    if os.path.exists(BIKE):
        d = np.load(BIKE)
        out.append(("Bike-hourly", d["X"], d["y"], "regression"))
    return out


def split(X, y, seed, clf):
    return train_test_split(np.asarray(X, np.float64), np.asarray(y),
                            test_size=0.2, random_state=seed,
                            stratify=(y if clf else None))


def reg_metrics(yte, yhat):
    return {"R2": float(r2_score(yte, yhat)),
            "MSE": float(mean_squared_error(yte, yhat)),
            "MAD": float(mean_absolute_error(yte, yhat))}


def clf_metrics(yte, pred, proba, classes):
    p1 = proba[:, 1]                                   # P(positive class); binary
    return {"ACC": float(accuracy_score(yte, pred)),
            "AUC": float(roc_auc_score(yte, p1)),
            "LogLoss": float(log_loss(yte, proba, labels=classes)),
            "Brier": float(brier_score_loss(yte, p1))}


def aggregate(per_seed, keys):
    agg = {}
    for k in keys:
        v = np.array([d[k] for d in per_seed])
        agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}
    return agg


def fmt(agg, keys):
    return "  ".join(f"{k}={agg[k]['mean']:.4f}+/-{agg[k]['std']:.4f}" for k in keys)


def run_msskm():
    from skm import MSSKM
    results = {}
    for name, X, y, task in datasets():
        clf = task == "classification"
        keys = CLF_METRICS if clf else REG_METRICS
        H, K = PARAMS[name]
        per_seed = []
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = split(X, y, seed, clf)
            t0 = time.time()
            m = MSSKM(task=task, H=H, K=K, seed=seed, verbose=False).fit(Xtr, ytr)
            if clf:
                met = clf_metrics(yte, m.predict(Xte), m.predict_proba(Xte), m.classes_)
            else:
                met = reg_metrics(yte, m.predict(Xte))
            per_seed.append(met)
            print(f"  [msskm] {name:<14} seed={seed} H={H} K={K}  "
                  f"{'  '.join(f'{k}={met[k]:.4f}' for k in keys)}  ({time.time()-t0:.0f}s)", flush=True)
        agg = aggregate(per_seed, keys)
        results[name] = {"model": "msskm", "H": H, "K": K, "task": task, "metrics": agg}
        print(f"== [msskm] {name}: {fmt(agg, keys)}", flush=True)
    return results


def run_catboost(n_trials):
    import optuna
    from catboost import CatBoostRegressor, CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = {}
    for name, X, y, task in datasets():
        clf = task == "classification"
        keys = CLF_METRICS if clf else REG_METRICS
        per_seed = []
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = split(X, y, seed, clf)
            Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed,
                                                    stratify=(ytr if clf else None))
            t0 = time.time()

            def make(params, iters):
                common = dict(iterations=iters, random_seed=seed, task_type="CPU",
                              thread_count=8, verbose=False, **params)
                return (CatBoostClassifier(loss_function="Logloss", **common) if clf
                        else CatBoostRegressor(loss_function="RMSE", **common))

            def objective(trial):
                params = dict(
                    depth=trial.suggest_int("depth", 4, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                    border_count=trial.suggest_categorical("border_count", [128, 254]),
                )
                mm = make(params, 2000)
                mm.fit(Xtr2, ytr2, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=False)
                trial.set_user_attr("best_iter", int(mm.get_best_iteration()))
                return (log_loss(yva, mm.predict_proba(Xva)) if clf
                        else float(np.sqrt(mean_squared_error(yva, mm.predict(Xva)))))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_iter = max(50, study.best_trial.user_attrs["best_iter"])
            final = make(dict(study.best_params), best_iter)
            final.fit(Xtr, ytr, verbose=False)
            if clf:
                met = clf_metrics(yte, final.predict(Xte).ravel(), final.predict_proba(Xte), final.classes_)
            else:
                met = reg_metrics(yte, final.predict(Xte))
            per_seed.append(met)
            print(f"  [catboost] {name:<14} seed={seed}  "
                  f"{'  '.join(f'{k}={met[k]:.4f}' for k in keys)}  ({time.time()-t0:.0f}s)", flush=True)
        agg = aggregate(per_seed, keys)
        results[name] = {"model": "catboost", "n_trials": n_trials, "task": task, "metrics": agg}
        print(f"== [catboost] {name}: {fmt(agg, keys)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["msskm", "catboost"], required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run_msskm() if args.model == "msskm" else run_catboost(args.trials)
    out = args.out or f"benchmarks/tenseed_{args.model}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
