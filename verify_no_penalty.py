"""
verify.py
=========

Side-by-side comparison of your from-scratch logistic regression vs sklearn's.

This is the test that tells you whether your implementation is correct.
Same data, same split, same scaling. You should be within 0.5% test accuracy
of sklearn.

Run AFTER train.py works:
    python verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from logreg_scratch import forward, predict, train, train_newton  # noqa: E402

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
X_train = scaler.fit_transform(X_train).astype(np.float64)
X_val = scaler.transform(X_val).astype(np.float64)
X_test = scaler.transform(X_test).astype(np.float64)

# ---------------------------------------------------------------------------
# 2. Train your from-scratch model
# ---------------------------------------------------------------------------
print("Training from-scratch model...")
scratch_w, scratch_b, _, _ = train_newton(
    X_train,
    y_train,
    X_val,
    y_val,
    lambda_=0.00,
    seed=42,
    verbose=False,
    max_iters=200,
    tol=1e-6,
)
scratch_proba = forward(X_test, scratch_w, scratch_b)
scratch_pred = predict(X_test, scratch_w, scratch_b, threshold=0.5)

# ---------------------------------------------------------------------------
# 3. Train sklearn's logistic regression (same hyperparameters as closely as possible)
# ---------------------------------------------------------------------------
print("Training sklearn model...")
# sklearn's C = 1 / lambda_; we used lambda_=0.01 so C=100
# lbfgs is the default solver and uses full-batch optimisation
# max_iter=1000 to ensure convergence
sklearn_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
sklearn_model.fit(X_train, y_train)
sklearn_proba = sklearn_model.predict_proba(X_test)[:, 1]
sklearn_pred = sklearn_model.predict(X_test)

# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Side-by-side comparison (test set)")
print("=" * 60)
print(f"{'Metric':<20}{'Scratch':>12}{'sklearn':>12}{'Diff':>12}")
print("-" * 60)

scratch_acc = accuracy_score(y_test, scratch_pred)
sklearn_acc = accuracy_score(y_test, sklearn_pred)
scratch_auc = roc_auc_score(y_test, scratch_proba)
sklearn_auc = roc_auc_score(y_test, sklearn_proba)
scratch_brier = brier_score_loss(y_test, scratch_proba)
sklearn_brier = brier_score_loss(y_test, sklearn_proba)

print(f"{'Accuracy':<20}{scratch_acc:>12.4f}{sklearn_acc:>12.4f}{scratch_acc - sklearn_acc:>+12.4f}")
print(f"{'AUC':<20}{scratch_auc:>12.4f}{sklearn_auc:>12.4f}{scratch_auc - sklearn_auc:>+12.4f}")
print(f"{'Brier score':<20}{scratch_brier:>12.4f}{sklearn_brier:>12.4f}{scratch_brier - sklearn_brier:>+12.4f}")

# Compare weights
print("\n" + "=" * 60)
print("Parameter comparison")
print("=" * 60)
print(f"{'Feature':<20}{'Scratch w':>15}{'sklearn w':>15}")
print("-" * 60)
feature_names = ["variance", "skewness", "curtosis", "entropy"]
for i, name in enumerate(feature_names):
    print(f"{name:<20}{scratch_w[i]:>15.4f}{sklearn_model.coef_[0][i]:>15.4f}")
print(f"{'(bias)':<20}{scratch_b:>15.4f}{sklearn_model.intercept_[0]:>15.4f}")

# Direction check: cosine similarity between the two weight vectors
import math
cos_sim = float(np.dot(scratch_w, sklearn_model.coef_[0]) / (np.linalg.norm(scratch_w) * np.linalg.norm(sklearn_model.coef_[0])))
angle_deg = math.degrees(math.acos(np.clip(cos_sim, -1.0, 1.0)))
print()
print(f"Cosine similarity: {cos_sim:.4f}")
print(f"Angle between weight vectors: {angle_deg:.2f} degrees")

# Sign check: do all weights have the same sign in both models?
signs_match = bool(np.all(np.sign(scratch_w) == np.sign(sklearn_model.coef_[0])))
print(f"Weight signs match: {signs_match}")

# ---------------------------------------------------------------------------
# 5. Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Verdict")
print("=" * 60)
acc_diff = abs(scratch_acc - sklearn_acc)

if acc_diff < 0.001 and cos_sim > 0.999 and signs_match:
    print("PASS - your implementation matches sklearn within tolerance (no regularisation).")
    print("  (Test acc diff < 0.001, cosine similarity > 0.999, all weight signs match)")
else:
    print("CHECK - at least one of the following is off:")
    print(f"  test accuracy diff: {acc_diff:.4f}  (target < 0.001)")
    print(f"  cosine similarity:  {cos_sim:.4f}  (target > 0.999)")
    print(f"  weight signs match: {signs_match}")
    print("  Things to check (debug gradient first): gradient formula, vectorisation, loss clipping, then L2/regularisation terms.")