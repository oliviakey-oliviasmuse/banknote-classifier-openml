"""
logreg_scratch.py
=================

A from-scratch implementation of binary logistic regression using only NumPy.

You implement the seven functions below. Each has a docstring that says what
it should do, a math hint, and a `# TODO(you)` line. The math is also in the
README.md at the project root — read that first.

Verification: run `python verify.py` to compare against sklearn's implementation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Sigmoid
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid: sigma(z) = 1 / (1 + exp(-z))."""
    return 1 / (1 + np.exp(-z))



# ---------------------------------------------------------------------------
# 2. Initialise parameters
# ---------------------------------------------------------------------------
def initialize_parameters(n_features: int, seed: int = 42) -> tuple[np.ndarray, float]:
    """Initialise weights and bias. Returns (w, b)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=0.01, size=n_features).astype(np.float64)
    b = 0.0
    return w, b


# ---------------------------------------------------------------------------
# 3. Forward pass (predictions)
# ---------------------------------------------------------------------------
def forward(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute predicted probabilities for a batch. Returns y_hat, shape (n,)."""
    z = X @ w + b 
    return sigmoid(z)



# ---------------------------------------------------------------------------
# 4. Compute loss (binary cross-entropy + L2)
# ---------------------------------------------------------------------------
def compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lambda_: float = 0.01,
    eps: float = 1e-15,
) -> float:
    """Binary cross-entropy loss with L2. Returns scalar."""
    y_hat = forward(X, w, b)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + (lambda_ / 2) * np.sum(w ** 2)
    return loss 


# ---------------------------------------------------------------------------
# 5. Compute gradients
# ---------------------------------------------------------------------------
def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lambda_: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Gradients of cost w.r.t. w and b. Returns (grad_w, grad_b)."""
    n_samples = X.shape[0]
    y_hat = forward(X, w, b)
    grad_w = X.T @ (y_hat - y) + lambda_ * w
    grad_b = np.sum(y_hat - y)
    return grad_w, grad_b


# ---------------------------------------------------------------------------
# 6. Training loop (mini-batch SGD with early stopping)
# ---------------------------------------------------------------------------
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.1,
    lambda_: float = 0.01,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 50,
    seed: int = 42,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[float], list[float]]:
    """Train with mini-batch SGD + early stopping. Returns (w, b, train_hist, val_hist)."""
    n_samples, n_features = X_train.shape
    w, b = initialize_parameters(n_features, seed)
    m_w, m_b = np.zeros_like(w), 0.0  # First moments (Adam)
    v_w, v_b = np.zeros_like(w), 0.0  # Second moments (Adam)
    train_hist, val_hist = [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    t = 0 # Step counter for Adam's bias correction

    for each in range(max_epochs):
        # Shuffle the training data
        rng = np.random.default_rng(seed + each)
        indices = rng.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch training
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            grad_w, grad_b = compute_gradients(X_batch, y_batch, w, b)
            t += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w**2
            w -= learning_rate * m_w / (np.sqrt(v_w) + eps)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b**2
            b -= learning_rate * m_b / (np.sqrt(v_b) + eps)

        # Compute and store training loss
        train_loss = compute_loss(X_train, y_train, w, b, lambda_)
        train_hist.append(train_loss)

        # Compute and store validation loss
        val_loss = compute_loss(X_val, y_val, w, b, lambda_)
        val_hist.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {each + 1}.")
                return w, b, train_hist, val_hist

        if verbose:
            print(f"Epoch {each + 1} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    return w, b, train_hist, val_hist

def train_newton(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.1,   # only used for the bias update
    lambda_: float = 0.01,
    max_iters: int = 50,
    tol: float = 1e-6,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[float], list[float]]:
    """Train with Newton's method. Returns (w, b, train_hist, val_hist)."""
    n_samples, n_features = X_train.shape
    w, b = initialize_parameters(n_features, seed)
    train_hist, val_hist = [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for each in range(max_iters):
        # Compute gradients and Hessian
        y_hat = forward(X_train, w, b)
        grad_w, grad_b = compute_gradients(X_train, y_train, w, b, lambda_)
        R = np.diag(y_hat * (1 - y_hat))
        H_w = X_train.T @ R @ X_train + lambda_ * np.eye(n_features)
        H_b = np.sum(R)

        # Newton step
        delta_w = np.linalg.solve(H_w, grad_w)
        delta_b = grad_b / H_b if H_b > 0 else 0.0
        w = w - delta_w
        b = b - delta_b

        # Compute and store training loss
        train_loss = compute_loss(X_train, y_train, w, b, lambda_)
        train_hist.append(train_loss)

        # Compute and store validation loss
        val_loss = compute_loss(X_val, y_val, w, b, lambda_)
        val_hist.append(val_loss)

        if verbose:
            print(f"iter {each + 1} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, |delta_w|={np.linalg.norm(delta_w):.6f}")

        # Convergence check on Newton step size
        if np.linalg.norm(delta_w) < tol:
            if verbose:
                print(f"Converged at iteration {each + 1}.")
            break

    return w, b, train_hist, val_hist


# ---------------------------------------------------------------------------
# 7. Prediction at a threshold
# ---------------------------------------------------------------------------
def predict(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict class labels at a probability threshold. Returns int array of {0, 1}."""
    y_hat = forward(X, w, b)
    return (y_hat >= threshold).astype(np.int64)
    
