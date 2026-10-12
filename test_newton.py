"""Quick comparison: train_newton vs sklearn on banknote."""
import numpy as np
from logreg_scratch import train_newton, forward, predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

data = np.genfromtxt(
    str(Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "banknote.csv"),,
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

# Newton
w, b, h_tr, h_va = train_newton(
    X_train, y_train, X_val, y_val, lambda_=0.01, max_iters=50, verbose=True
)
y_proba = forward(X_test, w, b)
y_pred = predict(X_test, w, b, 0.5)
print()
print(f"NEWTON  acc={accuracy_score(y_test, y_pred):.4f}  "
      f"AUC={roc_auc_score(y_test, y_proba):.4f}  "
      f"w={w}  b={b:.4f}")
print(f"        train loss final={h_tr[-1]:.4f}  val loss final={h_va[-1]:.4f}  "
      f"iters={len(h_tr)}")

# sklearn
m = LogisticRegression(C=100, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
m.fit(X_train, y_train)
print(f"SKLEARN acc={accuracy_score(y_test, m.predict(X_test)):.4f}  "
      f"AUC={roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.4f}  "
      f"w={m.coef_[0]}  b={m.intercept_[0]:.4f}")
