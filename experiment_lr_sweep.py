"""
experiment_lr_sweep.py
======================

SGD learning rate sweep on banknote (raw features, no scaling).

Tries 4 learning rates [0.001, 0.01, 0.1, 1.0] for a fixed budget
of 100 epochs. Plots loss vs epoch for all four on one chart
(log y-axis) and labels each as too slow / good / diverged.
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
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

# Run the LR sweep 
learning_rates = [0.001, 0.01, 0.1, 1.0]
results = []
for lr in learning_rates:
    w, b, tr_hist, va_hist = train(
        X_train, y_train, X_val, y_val,
        learning_rate=lr,
        lambda_=0.0,
        batch_size=32,
        max_epochs=100,
        patience=100, #disable early stopping
        seed=42,
        verbose=False
    )
    results.append({
        "lr": lr,
        "tr_hist": tr_hist,
        "va_hist": va_hist,
    })
    print(f"lr={lr}: {len(tr_hist)} epochs, final train loss={tr_hist[-1]:.4f}, final val loss={va_hist[-1]:.4f}")

# Plot the loss curves
fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    epochs = list(range(1, len(r["tr_hist"]) + 1))
    ax.semilogy(epochs, r["tr_hist"], label=f"lr={r['lr']}", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (log scale)")
ax.set_title("SGD learning rate sweep on banknote")
ax.legend(title="Learning rate", loc="best")
ax.grid(True, alpha=0.3)

fig_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
out_path = fig_dir / "lr_sweep.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved loss curves to {out_path}")
