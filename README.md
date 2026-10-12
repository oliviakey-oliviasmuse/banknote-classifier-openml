# Banknote Authentication: Logistic Regression from Scratch

A pure-NumPy implementation of binary logistic regression, trained with Newton's method, and benchmarked end-to-end against scikit-learn. Built as a foundation for understanding classical ML before moving to neural networks and conformal prediction.

---

## Problem

Banknote forgery detection is a binary classification task: given four numerical features extracted from a wavelet-transformed image of a banknote, predict whether the note is genuine or forged. This is a well-suited test case for an ML-from-scratch project because:

- The dataset is small (1372 samples) and well-understood
- The features are continuous and low-dimensional (4 features)
- The classification is essentially binary with a clear signal-to-noise story
- The same problem class (fraud detection, quality control) appears across capital-intensive industries

The goal: build logistic regression from first principles, train it, and characterise its behaviour thoroughly — including where it fails, where the standard library beats it, and what design choices (loss function, optimiser, threshold, regularisation) actually matter.

---

## Data

**Source:** OpenML banknote authentication dataset (id 1462), cached at `data/raw/banknote.csv`.

**Features (4):**

| Feature | Description |
|---|---|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Kurtosis of the wavelet-transformed image |
| `entropy` | Entropy of the wavelet-transformed image |

**Target:** `class` — 0 (genuine) or 1 (forged).

**Class balance:** 610 genuine, 762 forged (slightly imbalanced, 44%/56%).

**Split:** Stratified 80/20 train/test (random_state=42). The training set is further split 80/20 into train/val for early stopping. Fixed random seeds throughout for reproducibility.

---

## Method

### Model

Binary logistic regression. The model predicts:

```
P(y=1 | x) = σ(w·x + b)
```

where `σ` is the sigmoid function. Trained by minimising binary cross-entropy loss plus optional L2 regularisation.

### Implementation

The model is implemented in `logreg_scratch.py` — about 150 lines of pure NumPy. Seven functions:

| Function | What it does |
|---|---|
| `sigmoid` | Element-wise logistic sigmoid |
| `initialize_parameters` | Small random weight init, zero bias |
| `forward` | Linear + sigmoid prediction |
| `compute_loss` | Binary cross-entropy + L2 |
| `compute_gradients` | Gradient of loss w.r.t. weights and bias |
| `train` | Mini-batch SGD with early stopping |
| `predict` | Threshold-based class prediction |

Plus a bonus `train_newton` function that uses Newton's method (Hessian-based optimisation) — the key to matching sklearn's accuracy.

### Why Newton's method

The project went through 9 failed optimiser experiments before finding the right one. Vanilla SGD, SGD+momentum, and Adam all settled in different local optima, each with a ~1-3% accuracy gap to sklearn. Newton's method — a second-order optimiser that uses the Hessian (matrix of second derivatives) — closed the gap completely. For small problems like this one (4 features, 877 training samples), the Hessian is a tiny 4×4 matrix, so Newton's method is both fast and exact.

### Validation strategy

- Single 80/20 train/test split for the primary results
- K=5 and K=10 stratified cross-validation for robustness estimates
- A separate "no-penalty" verify (no L2) to test the gradient against sklearn using cosine similarity

### Threshold tuning

The default classification threshold of 0.5 is not appropriate for fraud detection, where the cost of a false negative (a forged note accepted as genuine) is much higher than the cost of a false positive (a genuine note rejected). The threshold is tuned to 0.3 to minimise false negatives at acceptable cost in false positives.

---

## Results

The headline result: the from-scratch Newton classifier achieves **98.55% test accuracy** at the tuned threshold, beating scikit-learn's default (97.45%) and the no-information baseline (55.64%).

See `results_table.md` for one row per experiment (17 total) and `experiments.md` for the full write-ups.

### Key findings

**The Newton breakthrough.** The model trained successfully with all optimisers, but only Newton's method matched scikit-learn exactly. The Hessian-based step size was the difference.

**Gradient correctness.** With no L2 regularisation, the weight vector has cosine similarity 1.0000 with sklearn's, and all weight signs match. The gradient implementation is correct, not just lucky.

**K-fold robustness.** 5-fold CV: 0.9905 ± 0.0067. 10-fold CV: 0.9920 ± 0.0106. The model's true accuracy is around 0.99 with about ±0.01 uncertainty.

**L2 doesn't help here.** All four regularisation strengths (0, 0.01, 0.1, 1.0) give identical test accuracy. L2 shrinks weights by up to 58% but doesn't change the decision boundary. The unregularised model is already at the optimal generalisation point.

**Threshold matters for asymmetric-cost domains.** At threshold 0.3, false negatives drop from 5 to 2 (catches 3 more forgeries) at no cost in false positives. F1 improves from 0.9769 to 0.9869.

**Beats the no-information baseline by 42.91 percentage points.** The model is 1.77x more accurate than just predicting "forged" for every note. It is extracting real signal from the features, not just a sophisticated coin flip.

### Headline numbers

| Metric | Value |
|---|---|
| Test accuracy (Newton, threshold 0.5) | 0.9745 |
| Test accuracy (Newton, threshold 0.3) | **0.9855** |
| AUC | 0.9990 |
| Brier score | 0.0153 |
| K-fold mean (5-fold) | 0.9905 ± 0.0067 |
| K-fold mean (10-fold) | 0.9920 ± 0.0106 |
| Sklearn reference (LogReg C=100, lbfgs) | 0.9745 |
| Majority-class baseline | 0.5564 |

---

## Limitations

**The dataset is nearly linearly separable.** The four features (variance, skewness, curtosis, entropy) of wavelet-transformed banknote images separate forged from genuine with very little overlap. As a result, test accuracy of 97-99% is expected from any reasonable classifier — including the simplest logistic regression. The model is not doing anything that a more sophisticated algorithm couldn't do on a harder dataset. **High accuracy is expected on this problem, and is not a strong signal of model quality in isolation.** A robust evaluation requires the baseline-comparison and cross-validation work documented in `results_table.md`.

**Small sample size.** 1372 notes is a small dataset by modern ML standards. The cross-validation estimates have meaningful uncertainty (~±0.01), and the model is unlikely to generalise to truly novel forgery techniques (which would have different feature distributions).

**Single dataset, single domain.** The work is on OpenML banknote authentication only. It does not generalise to credit card fraud, insurance fraud, or other "fraud-like" tasks without re-validation. The methodology transfers, but the model itself does not.

**Logistic regression assumptions.** The model assumes a linear decision boundary and no feature interactions. The banknote data happens to satisfy both. A more complex problem would need feature engineering (interactions, polynomials) or a non-linear model.

**Class imbalance is mild but not addressed.** The dataset is 44/56 genuine/forged. The model is not calibrated for highly imbalanced cases. SMOTE, class weighting, or threshold tuning per-class would be needed for a 99/1 imbalance.

**No conformal prediction.** The model gives point predictions but no calibrated uncertainty intervals. For high-stakes decisions (e.g., automatic rejection of a note), conformal prediction would provide per-prediction confidence guarantees.

---

## Next Steps

**Move to harder problems.** The banknote dataset is too easy. The next iteration should use a dataset with overlapping classes — credit card fraud (ULB ML Repository), Customer Churn (Kaggle), or a real industrial sensor anomaly detection problem.

**Implement conformal prediction.** The banknote project is the right sandbox for learning conformal prediction. It wraps any classifier and provides per-prediction confidence intervals with distribution-free guarantees. The methodology would be: implement `train_conformal(model, X_cal, y_cal)` that takes a fitted model and a calibration set, returns a `predict_conformal(X_test, alpha=0.1)` that returns prediction sets with 90% coverage.

**Move to softmax regression (multi-class).** Three-line change from the current binary implementation. The natural next project.

**Move to a single-hidden-layer neural net.** Same code, but the sigmoid becomes a ReLU and one more layer is added. The Newton method no longer applies (Hessian too big), so this will exercise Adam/momentum properly.

**Apply conformal prediction to operational tooling.** The methodology developed here generalises to flagging equipment failures, detecting quality defects, and identifying safety violations — all of which have the same asymmetric cost structure as fraud detection.

**Benchmark on harder classification tasks.** Once conformal prediction is built, benchmark on a hard dataset (e.g., 10-class image classification) to test the methodology in a regime where the simple logistic regression baseline fails.

---

## Repository structure

```
learn/logistic_regression_scratch/
├── logreg_scratch.py           # The 7 functions + train_newton (pure NumPy)
├── train.py                    # Training script with diagnostic plots
├── verify.py                   # Verify against sklearn with L2
├── verify_no_penalty.py        # Verify against sklearn without L2 (gradient test)
├── experiment_scaling.py       # Feature scaling experiment
├── experiment_lr_sweep.py      # Learning rate sweep
├── experiment_kfold.py         # K-fold cross-validation
├── experiment_l2.py            # L2 regularisation sweep
├── experiment_threshold.py     # Threshold sweep + manual metrics
├── results_table.md            # One row per experiment
├── experiment_log.md           # Logging template with filled examples
└── experiments.md              # Full write-up of each experiment
```

## How to run

```bash
# From the project root
cd openml-banknote-classifier/learn/logistic_regression_scratch

# Run the primary verification (Newton vs sklearn with L2)
python verify.py

# Run the no-penalty gradient test
python verify_no_penalty.py

# Run any individual experiment
python experiment_kfold.py
python experiment_threshold.py
```

All experiments use the same 80/20 split (`random_state=42`) and the same train/val split, so results are directly comparable.
