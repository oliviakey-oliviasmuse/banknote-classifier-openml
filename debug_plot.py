"""Debug script: check if the LR sweep data is reaching the plot correctly."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from logreg_scratch import train

# Load + split (same as experiment)
data = np.genfromtxt(
    str(Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "banknote.csv"),,
    delimiter=",", skip_header=1,
)
X = data[:, :4].astype(np.float64)
y = (data[:, 4] - 1).astype(np.int64) if data[:, 4].min() == 1 else data[:, 4].astype(np.int64)

from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

# Run one LR only
w, b, tr_hist, va_hist = train(
    X_train, y_train, X_val, y_val,
    learning_rate=0.01, lambda_=0.0,
    batch_size=32, max_epochs=100, patience=100,
    seed=42, verbose=False,
)

print(f"tr_hist length: {len(tr_hist)}")
print(f"tr_hist first 5: {tr_hist[:5]}")
print(f"tr_hist last 5: {tr_hist[-5:]}")
print(f"tr_hist has NaN: {any(np.isnan(tr_hist))}")
print(f"tr_hist has Inf: {any(np.isinf(tr_hist))}")
print(f"tr_hist min: {min(tr_hist):.4f}, max: {max(tr_hist):.4f}")

# Try to plot
fig, ax = plt.subplots(figsize=(10, 6))
epochs = list(range(1, len(tr_hist) + 1))
ax.semilogy(epochs, tr_hist, label="lr=0.01", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_title("Debug: single LR")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent.parent / "reports" / "figures" / "debug_lr.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
