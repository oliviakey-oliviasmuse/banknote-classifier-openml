"""Verbose Newton test."""
import numpy as np
from logreg_scratch import train_newton
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt(
    r"C:\Users\olivi\.minimax-agent\projects\openml-banknote-classifier\data\raw\banknote.csv",
    delimiter=",", skip_header=1,
)
X = data[:, :4].astype(np.float64)
y_raw = data[:, 4].astype(np.int64)
y = (y_raw - 1).astype(np.int64) if y_raw.min() == 1 else y_raw

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

# Manually run a few iterations of Newton to see what's happening
from logreg_scratch import forward, compute_gradients, compute_loss
import numpy as np

n_samples, n_features = X_train.shape
rng = np.random.default_rng(42)
w = rng.normal(0.0, 0.01, size=n_features).astype(np.float64)
b = 0.0

for i in range(10):
    y_hat = forward(X_train, w, b)
    grad_w, grad_b = compute_gradients(X_train, y_train, w, b, 0.01)
    R = np.diag(y_hat * (1 - y_hat))
    H_w = X_train.T @ R @ X_train + 0.01 * np.eye(n_features)
    H_b = np.sum(R)

    delta_w = np.linalg.solve(H_w, grad_w)
    delta_b = grad_b / H_b  # Newton for bias
    w_new = w - delta_w
    b_new = b - delta_b

    train_loss = compute_loss(X_train, y_train, w, b, 0.01)
    print(f"iter {i}: train_loss={train_loss:.4f}, |grad_w|={np.linalg.norm(grad_w):.4f}, "
          f"|delta_w|={np.linalg.norm(delta_w):.4f}, w_sum={w.sum():.4f}, b={b:.4f}")
    w, b = w_new, b_new
