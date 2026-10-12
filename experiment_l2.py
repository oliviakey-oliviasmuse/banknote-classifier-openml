"""
experiment_l2.py
=====================
L2 regularisation sweep on the Newton classifier.
Trains Newton with lambda in {0, 0.01, 0.1, 1.0}, records per-feature
weight magnitudes, plots |w| vs lambda, and prints a verdict table.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logreg_scratch import train_newton, forward, predict  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# 1. Load + split (same 80/20 stratified split as verify.py)
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

feature_names = ["variance", "skewness", "curtosis", "entropy"]
lambdas = [0.0, 0.01, 0.1, 1.0]

# L2 regularisation sweep
results = []
for lambda_ in lambdas:
    w, b, _, _ = train_newton(
        X_train, y_train, X_val, y_val,
        lambda_=lambda_, max_iters=200, tol=1e-6,
        seed=42, verbose=False,
    )
    proba = forward(X_test, w, b)
    pred = predict(X_test, w, b, threshold=0.5)
    test_acc = accuracy_score(y_test, pred)
    results.append({
        "lambda_": lambda_,
        "w": w.copy(),
        "test_acc": test_acc,
        "weight_norm": float(np.linalg.norm(w)),
    })
    print(f"lambda={lambda_}: test_acc={test_acc:.4f}, |w|={np.linalg.norm(w):.4f}")

    # Plot per-feature weight magnitudes vs lambda
fig, ax = plt.subplots(figsize=(10, 6))
W = np.array([np.abs(r["w"]) for r in results])  # shape (K, 4)
for i, name in enumerate(feature_names):
    ax.semilogx(lambdas, W[:, i], "-o", label=name, linewidth=1.5, markersize=8)
ax.set_xlabel("lambda (L2 strength, log scale)")
ax.set_ylabel("|w| (absolute weight magnitude)")
ax.set_title("Per-feature weight magnitude vs L2 strength (Newton on banknote)")
ax.legend(title="Feature", loc="best")
ax.grid(True, alpha=0.3)

fig_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
out_path = fig_dir / "l2_weights.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")