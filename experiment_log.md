# Experiment Log

This document provides a template for logging ML experiments in a consistent, comparable format. One row per **run** (a single execution of the training script with a particular configuration). One experiment may produce several rows (a hyperparameter sweep is several runs).

## The template

| Field | Description |
|---|---|
| `run_id` | Unique identifier (date + sequence number, e.g. `2026-10-08_03`) |
| `hypothesis` | What the runner expected to happen and why |
| `change` | What was different from the previous run (hyperparameter, code, data) |
| `metric_before` | The relevant metric value in the run just before this one (or "n/a" for first run) |
| `metric_after` | The relevant metric value in this run |
| `metric_name` | Which metric is being tracked (accuracy, F1, weight_diff, etc.) |
| `conclusion` | One sentence: was the hypothesis supported, and what was learned |

## When to use it

Every time a training script is run, add a row. The "change" column is the most important — it makes the log a record of what was tried, not just what worked. Even failed runs (where metric_after is worse) should be logged.

## Filled-in examples from this project

The 17 experiments in `results_table.md` were all logged in this format. Here are the most informative rows:

| run_id | hypothesis | change | metric_before | metric_after | metric_name | conclusion |
|---|---|---|---|---|---|---|
| 2026-10-08_01 | The from-scratch SGD model will match sklearn within 0.5% accuracy. | First run of `verify.py`, defaults (lr=0.1, batch=32, lambda=0.01). | n/a | 0.9600 | test_accuracy | **Hypothesis wrong.** 1.5% gap. Investigate. |
| 2026-10-08_05 | Newton's method will close the 1.5% gap because it uses second-order info. | Switched from `train` (SGD) to `train_newton`. lambda=0.01. | 0.9600 | **0.9745** | test_accuracy | **Hypothesis right.** Newton matches sklearn exactly. |
| 2026-10-09_03 | Adding L2 will improve generalisation (test accuracy). | L2 sweep: lambda in {0, 0.01, 0.1, 1.0}. | 0.9745 (no L2) | 0.9745 (all 4) | test_accuracy | **Hypothesis wrong.** L2 has no effect on this dataset. Decision boundary is unchanged. |
| 2026-10-12_01 | Lowering the threshold to 0.3 will catch more forgeries. | Threshold sweep: t in {0.3, 0.5, 0.7}. | FN=5, FP=2 (t=0.5) | FN=2, FP=2 (t=0.3) | confusion_matrix | **Hypothesis right.** Threshold 0.3 catches 3 more forgeries at no cost in false positives. |
| 2026-10-12_02 | The model's accuracy is mostly driven by predicting the majority class. | Computed majority-class baseline. | (assumed high) | baseline=0.5564, model=0.9855 | baseline_comparison | **Hypothesis wrong.** Model is 42.91 percentage points above baseline, 1.77x more accurate. Real learning. |

## Why this format works

**Hypothesis-first.** Forces the runner to predict *before* running. This is the single biggest source of learning. Without a written prediction, you can rationalise any outcome.

**Change-tracked.** The "change" column makes the log a complete record of the experiment. Anyone reading it can reconstruct the entire sequence of attempts.

**Metric-comparable.** Each row has a metric_before and metric_after, so you can read down the column and see how the metric evolved. The "metric_name" column is essential — without it, the same number could mean accuracy, F1, or something else.

**Conclusion-required.** The "conclusion" column forces a one-sentence summary. If you can't write the conclusion in one sentence, you don't understand what happened.

## When a hypothesis is wrong

A wrong hypothesis is **more valuable** than a right one. A right hypothesis confirms what you already believed. A wrong hypothesis forces you to update your mental model.

For example, the row `2026-10-09_03` records that L2 did not improve generalisation, even though it shrank weights by 58%. The conclusion ("Decision boundary is unchanged") is a more useful insight than "L2 doesn't help" because it explains *why* L2 didn't help — and that explanation generalises to other small, well-separated datasets.

## Anti-patterns

- **Logging only successful runs.** Failed experiments are data. If the change made things worse, log it. The conclusion column is where you say "this didn't work because…"
- **Vague hypotheses.** "It will work" is not a hypothesis. A good hypothesis is testable: "Changing X from A to B will improve metric Y by approximately Z, because…"
- **Missing the change column.** The change column is the most important. It tells future-you (or future-team-members) what you actually tried.
- **Comparing incomparable runs.** A run with a different random seed or different data split is not directly comparable. Note the random seed in the change column.
