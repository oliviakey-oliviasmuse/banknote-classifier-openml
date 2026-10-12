"""Smoke tests for all 7 functions. Run with: python test_all.py"""

import numpy as np
from logreg_scratch import (
    sigmoid,
    initialize_parameters,
    forward,
    compute_loss,
    compute_gradients,
    train,
    predict,
)

def t(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except NotImplementedError as e:
        print(f"  ----  {name}  (not implemented yet)")
    except AssertionError as e:
        print(f"  FAIL  {name}  -> {e}")
    except Exception as e:
        print(f"  ERR   {name}  -> {type(e).__name__}: {e}")


print("Smoke tests:")
print("-" * 60)

def test_sigmoid():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0], atol=1e-6), f"got {out}"
t("sigmoid", test_sigmoid)

def test_init():
    w, b = initialize_parameters(n_features=4, seed=42)
    assert w.shape == (4,), f"w shape {w.shape}"
    assert isinstance(b, float), f"b type {type(b)}"
t("initialize_parameters", test_init)

def test_forward():
    w, b = initialize_parameters(4, 42)
    X = np.random.randn(5, 4)
    y_hat = forward(X, w, b)
    assert y_hat.shape == (5,), f"shape {y_hat.shape}"
    assert np.all((y_hat > 0) & (y_hat < 1)), f"out of (0,1): {y_hat}"
t("forward", test_forward)

def test_loss():
    w, b = initialize_parameters(4, 42)
    X = np.random.randn(10, 4)
    y = np.random.randint(0, 2, 10)
    loss = compute_loss(X, y, w, b, lambda_=0.01)
    assert np.isfinite(loss), f"loss not finite: {loss}"
    assert loss > 0, f"loss must be positive: {loss}"
t("compute_loss", test_loss)

def test_grads():
    w, b = initialize_parameters(4, 42)
    X = np.random.randn(10, 4)
    y = np.random.randint(0, 2, 10)
    gw, gb = compute_gradients(X, y, w, b, lambda_=0.01)
    assert gw.shape == (4,), f"grad_w shape {gw.shape}"
    assert np.isfinite(gw).all(), f"grad_w not finite"
    assert np.isfinite(gb), f"grad_b not finite"
t("compute_gradients", test_grads)

def test_train():
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_tr, X_va, y_tr, y_va = X[:80], X[80:], y[:80], y[80:]
    w, b, tr_hist, va_hist = train(
        X_tr, y_tr, X_va, y_va,
        learning_rate=0.1, lambda_=0.01,
        batch_size=16, max_epochs=200, patience=30,
        seed=42, verbose=False,
    )
    assert w.shape == (4,), f"w shape {w.shape}"
    assert len(tr_hist) > 0, "no training history"
    assert tr_hist[-1] < tr_hist[0], f"loss did not decrease: {tr_hist[0]:.4f} -> {tr_hist[-1]:.4f}"
t("train", test_train)

def test_predict():
    w, b = initialize_parameters(4, 42)
    X = np.random.randn(5, 4)
    y_pred = predict(X, w, b, threshold=0.5)
    assert y_pred.shape == (5,), f"shape {y_pred.shape}"
    assert set(np.unique(y_pred)).issubset({0, 1}), f"unexpected values: {np.unique(y_pred)}"
t("predict", test_predict)

print("-" * 60)
print("Done. Functions with ---- need implementing.")
