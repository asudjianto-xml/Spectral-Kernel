"""Smoke test: the single-bank MS-SKM fits a regression and a classification task.

Run directly (``python tests/test_smoke.py``) or under pytest. Thresholds are
loose sanity floors, not benchmark targets -- they only assert the model learns.
"""
import sys
import os
import time

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skm import MSSKM, SpectralGAM, LearnedGAM


def test_learned_gam_regression_california():
    X, y = fetch_california_housing(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    t0 = time.time()
    m = LearnedGAM(task="regression", seed=0).fit(Xtr, ytr)
    r2 = m.score(Xte, yte)
    grid, fj = m.shape_function(0)
    print(f"[LearnedGAM california] test R2 = {r2:.4f}  ({time.time() - t0:.1f}s)")
    assert r2 > 0.60, f"LearnedGAM R2 too low: {r2}"
    assert abs(float(np.mean(fj))) < 1e-5, "shape function should be centered"


def test_learned_gam_classification_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    m = LearnedGAM(task="classification", seed=0).fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    print(f"[LearnedGAM breast_cancer] test acc = {acc:.4f}")
    assert acc > 0.90, f"LearnedGAM accuracy too low: {acc}"


def test_gam_regression_california():
    X, y = fetch_california_housing(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    t0 = time.time()
    m = SpectralGAM(task="regression", seed=0).fit(Xtr, ytr)
    r2 = m.score(Xte, yte)
    grid, fj = m.shape_function(0)
    print(f"[GAM california] test R2 = {r2:.4f}  ({time.time() - t0:.1f}s)  "
          f"shape f_0 over {grid.shape[0]} pts")
    assert r2 > 0.60, f"GAM R2 too low: {r2}"
    assert abs(float(np.mean(fj))) < 1e-6, "shape function should be centered"


def test_gam_classification_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    m = SpectralGAM(task="classification", seed=0).fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    print(f"[GAM breast_cancer] test acc = {acc:.4f}")
    assert acc > 0.90, f"GAM accuracy too low: {acc}"


def test_multibank_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    m = MSSKM(task="classification", H=4, epochs=120, patience=15, seed=0).fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    w = m.bank_weights_
    print(f"[multibank H=4 breast_cancer] test acc = {acc:.4f}  bank_w={np.round(w, 3)}")
    assert acc > 0.90, f"multi-bank accuracy too low: {acc}"
    assert w.shape == (4,) and abs(w.sum() - 1.0) < 1e-5, "fusion weights should be convex over 4 banks"


def test_regression_california():
    X, y = fetch_california_housing(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    t0 = time.time()
    m = MSSKM(task="regression", epochs=120, patience=15, seed=0).fit(Xtr, ytr)
    r2 = m.score(Xte, yte)
    print(f"[california] test R2 = {r2:.4f}  ({time.time() - t0:.1f}s)  "
          f"ard range [{m.ard_.min():.3f}, {m.ard_.max():.3f}]")
    assert r2 > 0.70, f"R2 too low: {r2}"


def test_classification_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    t0 = time.time()
    m = MSSKM(task="classification", epochs=120, patience=15, seed=0).fit(Xtr, ytr)
    acc = m.score(Xte, yte)
    proba = m.predict_proba(Xte)
    print(f"[breast_cancer] test acc = {acc:.4f}  ({time.time() - t0:.1f}s)  "
          f"proba shape {proba.shape}")
    assert acc > 0.90, f"accuracy too low: {acc}"
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)


if __name__ == "__main__":
    test_gam_regression_california()
    test_gam_classification_breast_cancer()
    test_learned_gam_regression_california()
    test_learned_gam_classification_breast_cancer()
    test_multibank_breast_cancer()
    test_regression_california()
    test_classification_breast_cancer()
    print("\nOK")
