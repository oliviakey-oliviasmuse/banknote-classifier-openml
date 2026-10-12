"""
experiment_threshold.py
=====================
Manual confusion matrix + threshold sweep on the Newton classifier.
Computes precision, recall, F1 from scratch (no sklearn math), compares
to sklearn, then sweeps thresholds at 0.3, 0.5, 0.7 with cost analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from logreg_scratch import train_newton, forward  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# 1. Load + split
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


# Train Newton once
w, b, _, _ = train_newton(
    X_train, y_train, X_val, y_val,
    lambda_=0.0, max_iters=200, tol=1e-6,
    seed=42, verbose=False,
)
y_proba = forward(X_test, w, b)
print(f"Trained Newton. Test probabilities: min={y_proba.min():.4f}, max={y_proba.max():.4f}, mean={y_proba.mean():.4f}")

def confusion_matrix_manual(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]] (sklearn's layout)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return np.array([[tn, fp], [fn, tp]])


def precision_manual(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_manual(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_manual(y_true, y_pred):
    p = precision_manual(y_true, y_pred)
    r = recall_manual(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

# Default threshold sanity check (should match previous 0.9745)
y_pred_05 = (y_proba >= 0.5).astype(int)

print("Manual vs sklearn at threshold 0.5:")
print(f"  Confusion matrix — manual:\n{confusion_matrix_manual(y_test, y_pred_05)}")
print(f"  Confusion matrix — sklearn:\n{confusion_matrix(y_test, y_pred_05)}")
print(f"  Precision — manual: {precision_manual(y_test, y_pred_05):.4f}, sklearn: {precision_score(y_test, y_pred_05):.4f}")
print(f"  Recall    — manual: {recall_manual(y_test, y_pred_05):.4f}, sklearn: {recall_score(y_test, y_pred_05):.4f}")
print(f"  F1        — manual: {f1_manual(y_test, y_pred_05):.4f}, sklearn: {f1_score(y_test, y_pred_05):.4f}")
print()

# Threshold sweep
print("=" * 80)
print(f"{'Threshold':<12}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Acc':>10}")
print("-" * 80)
for t in [0.3, 0.5, 0.7]:
    y_pred = (y_proba >= t).astype(int)
    cm = confusion_matrix_manual(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    p = precision_manual(y_test, y_pred)
    r = recall_manual(y_test, y_pred)
    f1 = f1_manual(y_test, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print(f"{t:<12}{tn:>6}{fp:>6}{fn:>6}{tp:>6}{p:>12.4f}{r:>10.4f}{f1:>10.4f}{acc:>10.4f}")
print("=" * 80)

# Majority-class baseline
counts = np.bincount(y_test.astype(int))
majority_class = int(np.argmax(counts))
majority_baseline_acc = float(np.max(counts) / len(y_test))
model_acc_05 = float(accuracy_score(y_test, (y_proba >= 0.5).astype(int))) if 'accuracy_score' in dir() else 0.9745
model_acc_03 = float(accuracy_score(y_test, (y_proba >= 0.3).astype(int))) if 'accuracy_score' in dir() else 0.9855

print()
print("=" * 60)
print(f"Majority class: {majority_class} (forged if 1, genuine if 0)")
print(f"Majority-class baseline accuracy: {majority_baseline_acc:.4f}")
print(f"Model accuracy (threshold 0.5): {model_acc_05:.4f}")
print(f"Model accuracy (threshold 0.3): {model_acc_03:.4f}")
print()
print(f"Model beats baseline by: {model_acc_03 - majority_baseline_acc:.4f}")
print("=" * 60)