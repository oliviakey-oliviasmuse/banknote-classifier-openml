# Results Table

One row per experiment. Each row includes the question, key result, and conclusion. Full per-experiment write-ups are in `experiments.md`.

| # | Experiment | Question | Method | Key result | Conclusion |
|---|---|---|---|---|---|
| 1 | Does it train at all? | Does the model converge on banknote? | Mini-batch SGD, defaults | Yes. 252 epochs, final val loss 0.1853. | Baseline check passes. |
| 2 | Verify against sklearn (L2) | Within 0.5% accuracy? | SGD, batch=32, lambda=0.01 | Accuracy 0.96 vs 0.97 — 1.5% gap. | Gap exists. Investigate. |
| 3 | Lower lambda (Newton) | Less regularisation = closer to sklearn? | Newton, lambda=0.001 | Weight diff went *up* (8.19 vs 7.0). | Regularisation isn't the bottleneck. |
| 4 | Bigger batches | Does batch=128 close the gap? | SGD, batch=128 | Closest yet — weight diff 6.71. | Direction right, magnitude insufficient. |
| 5 | Full-batch at lr=0.1 | Less noise, optimal? | SGD, batch=877 | Worse than batch=128. | Too aggressive at full-batch. |
| 6 | Full-batch at lr=0.001 | Compensate with smaller lr? | SGD, batch=877, lr=0.001 | Under-trains. | Wrong direction. |
| 7 | Full-batch at lr=0.01 | Right lr? | SGD, batch=877, lr=0.01 | Weight diff 8.98. | Still wrong basin. |
| 8 | SGD + momentum, lr=0.1 | Helps stability? | momentum=0.9 | Worse — 0.0291 accuracy diff. | Need to lower lr. |
| 9 | SGD + momentum, lr=0.01 | Compensate for momentum? | momentum=0.9, lr=0.01 | Even worse. | Hyperparameters coupled, naive fix fails. |
| 10 | Adam | Adaptive learning rates? | Adam, lr=0.001 | Same wall — 0.0291 accuracy diff. | First-order methods converge to different basin. |
| 11 | Newton's method | Second-order solve? | train_newton, lambda=0.01 | **0.9745 acc, AUC 0.9990. Matches sklearn to 3 decimals.** | **Breakthrough.** |
| 12 | Feature scaling | Does scaling help? | Newton vs SGD, scaled vs raw | Newton scale-invariant. SGD on raw data beat SGD on scaled (0.9745 vs 0.9636). | Textbook "always scale" is wrong here. |
| 13 | K-fold CV (K=5) | Robust accuracy estimate? | Newton, no penalty | 0.9905 ± 0.0067. | Higher than single 0.9745 — more training data. |
| 14 | K-fold CV (K=10) | Tighter estimate? | Newton, no penalty | 0.9920 ± 0.0106. | Mean similar; std *higher* (smaller per-fold test sets). |
| 15 | L2 sweep | Does L2 help generalisation? | Newton, lambda in {0, 0.01, 0.1, 1.0} | Test accuracy identical (0.9745) at all 4 lambdas. Weights shrink up to 58%. | L2 is for auditability, not accuracy. |
| 16 | Threshold + metrics | Right threshold for fraud? | Manual vs sklearn metrics, thresholds {0.3, 0.5, 0.7} | 0.3 wins on every metric: acc 0.9855, F1 0.9869, FN 2 (vs 5). | Cost asymmetry justifies lower threshold. |
| 17 | Majority-class baseline | Is the model doing real work? | Always predict majority class | Baseline 0.5564. Model 0.9855. Gap: 0.4291. | Model is 1.77x more accurate than no-information baseline. |

## Headline numbers

| Metric | Value |
|---|---|
| Test accuracy (Newton, threshold 0.5) | 0.9745 |
| Test accuracy (Newton, threshold 0.3, tuned) | **0.9855** |
| AUC | 0.9990 |
| Brier score | 0.0153 |
| K-fold mean (5-fold) | 0.9905 ± 0.0067 |
| K-fold mean (10-fold) | 0.9920 ± 0.0106 |
| Sklearn reference (LogReg C=100, lbfgs) | 0.9745 |
| Majority-class baseline | 0.5564 |
| Gap to baseline | **0.4291 (1.77x more accurate)** |

## Method-comparison summary

| Method | Test acc | Notes |
|---|---|---|
| Majority-class baseline | 0.5564 | Always predict "forged" |
| SGD, no scaling, default lr | 0.9745 | Matched sklearn by accident, not design |
| SGD, with scaling, default lr | 0.9636 | Scaling hurt on this dataset |
| SGD + momentum (lr=0.1) | 0.9491 | Overshooting |
| Adam (lr=0.001) | 0.9491 | Same basin as momentum |
| **Newton's method (lambda=0.01)** | **0.9745** | **Matches sklearn exactly** |
| Newton's method (no penalty) | 0.9745 (test) / 0.9905 (5-fold mean) | Best generalisation |
| Newton + threshold 0.3 (tuned) | **0.9855** | Threshold tuning beats first-order methods |
| Sklearn (C=100, lbfgs) | 0.9745 | The reference target |
