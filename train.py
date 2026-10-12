"""
train.py
========

End-to-end training script for your from-scratch logistic regression.

What this file does (already written for you):
    1. Loads the banknote dataset from your existing cache
    2. Stratified train/val/test split (same as your production pipeline)
    3. Standardises features
    4. Calls your train() function from logreg_scratch.py
    5. Evaluates on the test set
    6. Plots train/val loss curve
    7. Plots calibration curve

What you need to do:
    - Make sure logreg_scratch.py is fully implemented (run the smoke tests
      in each docstring first)
    - Tweak hyperparameters (learning_rate, lambda_, batch_size) and observe
    - Fill in experiments.md as you go

Run:
    python train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make the scratch module importable
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from logreg_scratch import compute_loss, predict, train  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Load the banknote dataset (OpenML 1462)
# ---------------------------------------------------------------------------
# Try to use your existing cached CSV first; fall back to OpenML fetch.
CACHE = HERE.parent.parent / "data" / "raw" / "banknote.csv"

if CACHE.exists():
    print(f"Loading from cache: {CACHE}")
    data = np.genfromtxt(CACHE, delimiter=",", skip_header=1)
    X = data[:, :4].astype(np.float64)
    y = data[:, 4].astype(np.int64)
    # OpenML returns class as {1, 2}; remap to {0, 1} (0 = genuine, 1 = forged)
    y = (y - 1).astype(np.int64) if y.min() == 1 else y
else:
    print("Cache not found, fetching from OpenML...")
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=1462, as_frame=False, parser="auto")
    X = ds.data.astype(np.float64)
    y = ds.target.astype(np.int64)
    # Remap class labels: OpenML 1462 uses {1, 2}, we want {0, 1}
    y = (y - 1).astype(np.int64) if y.min() == 1 else y

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class balance: {np.bincount(y)}  (0=genuine, 1=forged)")

# ---------------------------------------------------------------------------
# 2. Stratified split: 80% train+val, 20% test; then 80/20 train/val
# ---------------------------------------------------------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)
print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# ---------------------------------------------------------------------------
# 3. Standardise features (zero mean, unit variance)
#    IMPORTANT: fit the scaler on TRAIN only, then transform val and test.
#    This is the same leakage trap the banknote project has to be careful about.
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float64)
X_val = scaler.transform(X_val).astype(np.float64)
X_test = scaler.transform(X_test).astype(np.float64)
print(f"After scaling — train mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")

# ---------------------------------------------------------------------------
# 4. Train your from-scratch model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training from-scratch logistic regression...")
print("=" * 60)

best_w, best_b, train_loss_hist, val_loss_hist = train(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.01,
    lambda_=0.01,
    batch_size=877,
    max_epochs=500,
    patience=50,
    seed=42,
    verbose=True,
)

print(f"\nTraining stopped. Best val loss: {min(val_loss_hist):.4f}")
print(f"Epochs run: {len(train_loss_hist)}")
print(f"Final weights: {best_w}")
print(f"Final bias: {best_b:.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluate on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Test set evaluation")
print("=" * 60)

# Probabilities for calibration + AUC
from logreg_scratch import forward  # noqa: E402

y_test_proba = forward(X_test, best_w, best_b)
y_test_pred = predict(X_test, best_w, best_b, threshold=0.5)

acc = accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)
brier = brier_score_loss(y_test, y_test_proba)
ll = log_loss(y_test, np.clip(y_test_proba, 1e-15, 1 - 1e-15))
cm = confusion_matrix(y_test, y_test_pred)

print(f"Test accuracy:    {acc:.4f}")
print(f"Test AUC:         {auc:.4f}")
print(f"Test Brier score: {brier:.4f}")
print(f"Test log loss:    {ll:.4f}")
print(f"Confusion matrix:\n{cm}")

# ---------------------------------------------------------------------------
# 6. Plot train/val loss curve
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(train_loss_hist, label="train", linewidth=1.5)
axes[0].plot(val_loss_hist, label="val", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (binary cross-entropy + L2)")
axes[0].set_title("Training and validation loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# ---------------------------------------------------------------------------
# 7. Plot calibration curve
# ---------------------------------------------------------------------------
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_idx = np.digitize(y_test_proba, bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)

empirical_freq = np.zeros(n_bins)
predicted_mean = np.zeros(n_bins)
counts = np.zeros(n_bins)
for i in range(n_bins):
    mask = bin_idx == i
    if mask.sum() > 0:
        empirical_freq[i] = y_test[mask].mean()
        predicted_mean[i] = y_test_proba[mask].mean()
        counts[i] = mask.sum()

axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
axes[1].plot(predicted_mean, empirical_freq, "o-", markersize=8, label="model")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Empirical positive rate")
axes[1].set_title(f"Calibration (n_bins={n_bins})")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.tight_layout()

# Save figure
fig_dir = HERE.parent.parent / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
out_path = fig_dir / "scratch_logreg_diagnostics.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved diagnostic plot: {out_path}")

# Final summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Test accuracy: {acc:.4f}  (target: >= 0.98 on banknote)")
print(f"Test AUC:      {auc:.4f}  (target: >= 0.99)")
print(f"Brier score:   {brier:.4f}  (lower = better calibration)")
print("\nNext: run `python verify.py` to compare against sklearn.")
