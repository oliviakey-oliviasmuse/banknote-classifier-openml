"""
experiment_scaling.py
=====================

Compare Newton and SGD with vs without feature scaling.

Records: iterations to converge, final train/val loss, test accuracy.
The point: feature scaling has a much bigger effect on first-order
methods (SGD) than on second-order methods (Newton).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from logreg_scratch import train_newton, train, forward, predict, compute_loss  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# 1. Load + split (same as train.py — keep these in sync)
# ---------------------------------------------------------------------------
CACHE = HERE.parent.parent / "data" / "raw" / "banknote.csv"
data = np.genfromtxt(CACHE, delimiter=",", skip_header=1)
X = data[:, :4].astype(np.float64)
y = data[:, 4].astype(np.int64)
y = (y - 1).astype(np.int64) if y.min() == 1 else y

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float64)
X_val_s = scaler.transform(X_val).astype(np.float64)
X_test_s = scaler.transform(X_test).astype(np.float64)

def run_one(X_tr, y_tr, X_va, y_va, X_te, y_te, method, label):
    """Train with the given method, return a dict of metrics."""
    if method == "newton":
        w, b, tr_hist, va_hist = train_newton(
            X_tr, y_tr, X_va, y_va,
            lambda_=0.0, max_iters=200, tol=1e-6,
            seed=42, verbose=False,
        )
    else:  # sgd
        w, b, tr_hist, va_hist = train(
            X_tr, y_tr, X_va, y_va,
            learning_rate=0.1, lambda_=0.0,
            batch_size=32, max_epochs=500, patience=50,
            seed=42, verbose=False,
        )
    proba = forward(X_te, w, b)
    pred = predict(X_te, w, b, threshold=0.5)
    return {
        "method": method,
        "label": label,
        "iterations": len(tr_hist),
        "train_loss": float(tr_hist[-1]),
        "val_loss": float(va_hist[-1]),
        "test_accuracy": float(accuracy_score(y_te, pred)),
    }


# Run the four experiments
results = []
results.append(run_one(X_train, y_train, X_val, y_val, X_test, y_test, "newton", "Newton + raw"))
results.append(run_one(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, "newton", "Newton + scaled"))
results.append(run_one(X_train, y_train, X_val, y_val, X_test, y_test, "sgd", "SGD + raw"))
results.append(run_one(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, "sgd", "SGD + scaled"))

# Print the comparison table
print()
print("=" * 70)
print(f"{'Method':<20}{'Scaled':>8}{'Iters':>8}{'TrainLoss':>14}{'ValLoss':>10}{'TestAcc':>10}")
print("-" * 70)
for r in results:
    scaled = "yes" if "scaled" in r["label"] else "no"
    print(f"{r['label']:<20}{scaled:>8}{r['iterations']:>8}"
          f"{r['train_loss']:>14.2f}{r['val_loss']:>10.2f}{r['test_accuracy']:>10.4f}")
print("=" * 70)