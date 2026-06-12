"""Tuned CatBoost baseline on the MS-SKM sweep datasets, same test splits.

For each dataset we hold out the same 20% test split as benchmarks/sweep_hk.py
(test_size=0.2, random_state=0, stratified for classification), carve a validation
fold from the remaining train for early stopping, and tune CatBoost with Optuna
(depth, learning rate, L2, bagging, random strength) over N_TRIALS. The best
configuration is refit on the full train fold at its early-stopped iteration count
and evaluated once on the held-out test set. Metric: R^2 (regression), accuracy
(classification) -- identical to the MS-SKM sweep.

Runs on CPU to avoid contending with a GPU sweep.

    python benchmarks/catboost_baseline.py
"""
import os
import time
import warnings

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, log_loss, mean_squared_error
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 0
N_TRIALS = 60
BIKE = os.path.expanduser("~/jupyterlab/multi-kernel/mnca/bike_hourly.npz")


def tune_eval(name, X, y, task):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    clf = task == "classification"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                          stratify=(y if clf else None))
    Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED,
                                             stratify=(ytr if clf else None))

    def make(params, iters=2000):
        common = dict(iterations=iters, random_seed=SEED, task_type="CPU",
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
        m = make(params)
        m.fit(Xtr2, ytr2, eval_set=(Xva, yva), early_stopping_rounds=50, verbose=False)
        trial.set_user_attr("best_iter", int(m.get_best_iteration()))
        if clf:
            return log_loss(yva, m.predict_proba(Xva))
        return float(np.sqrt(mean_squared_error(yva, m.predict(Xva))))

    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = dict(study.best_params)
    best_iter = max(50, study.best_trial.user_attrs["best_iter"])

    # refit on the full train fold at the tuned iteration count, evaluate on test
    final = make(best, iters=best_iter)
    final.fit(Xtr, ytr, verbose=False)
    metric = (accuracy_score(yte, final.predict(Xte)) if clf
              else r2_score(yte, final.predict(Xte)))
    mname = "acc" if clf else "R2"
    print(f"{name:<22} CatBoost {mname}={metric:.4f}  "
          f"(depth={best['depth']}, lr={best['learning_rate']:.3f}, "
          f"l2={best['l2_leaf_reg']:.1f}, iters={best_iter}, {N_TRIALS} trials, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return metric


def main():
    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    Xc, yc = fetch_california_housing(return_X_y=True)
    tune_eval("California (R2)", Xc, yc, "regression")
    Xb, yb = load_breast_cancer(return_X_y=True)
    tune_eval("Breast cancer (acc)", Xb, yb, "classification")
    if os.path.exists(BIKE):
        d = np.load(BIKE)
        tune_eval("Bike-hourly (R2)", d["X"], d["y"], "regression")
    else:
        print(f"(bike not found at {BIKE}; skipped)")


if __name__ == "__main__":
    main()
