"""
experiment_kfold.py
=====================
K-fold cross-validation on the banknote classifier (Newton).
Uses 5-fold stratified CV, reports per-fold accuracy and mean +/- std.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from logreg_scratch import train_newton, forward, predict  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
CACHE = HERE.parent.parent / "data" / "raw" / "banknote.csv"
data = np.genfromtxt(CACHE, delimiter=",", skip_header=1)
X = data[:, :4].astype(np.float64)
y = data[:, 4].astype(np.int64)
y = (y - 1).astype(np.int64) if y.min() == 1 else y


# K-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    w, b, _, _ = train_newton(
        X_train, y_train, X_test, y_test,
        lambda_=0.0, max_iters=200, tol=1e-6,
        seed=42, verbose=False,
    )
    proba = forward(X_test, w, b)
    pred = predict(X_test, w, b, threshold=0.5)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"Fold {fold_idx}: accuracy = {acc:.4f}")

# Summary
mean_acc = float(np.mean(accuracies))
std_acc = float(np.std(accuracies))
print()
print("=" * 50)
print(f"Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
print(f"Min: {min(accuracies):.4f}, Max: {max(accuracies):.4f}")
print("=" * 50)