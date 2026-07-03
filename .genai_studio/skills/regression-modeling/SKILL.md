---
name: regression-modeling
description: Fit and evaluate a predictive model honestly (regression or classification). Use when asked to predict, model, forecast, or find what drives an outcome. Splits train/test to avoid leakage, fits with the tools, evaluates on held-out data with the right metric, and reports honest performance + what the model says — never a fabricated accuracy.
---

# Regression / classification modeling

Use `fit_model` / `python_exec` (sklearn) for everything. The cardinal sin is reporting performance the
model didn't actually earn — **evaluate on data the model never saw**, and never invent a metric.

## The workflow
1. **Frame it** — regression (numeric target) or classification (categorical)? Name the target and the
   features. Check the target's distribution / class balance first.
2. **Guard against leakage** — drop features that trivially encode the target or wouldn't exist at
   prediction time; handle IDs/dates deliberately.
3. **Split** — a held-out test set (or cross-validation) BEFORE fitting. Fit preprocessing (scaling,
   imputation, encoding) on TRAIN only, then apply to test (a Pipeline prevents leakage).
4. **Baseline first** — a trivial baseline (mean/median for regression, majority-class for
   classification). A model only earns its keep by beating the baseline.
5. **Fit** a sensible model (start simple: linear/logistic; then a tree/forest if needed).
6. **Evaluate on the held-out set with the RIGHT metric**:
   - regression: RMSE / MAE + R^2 (vs the baseline).
   - classification: accuracy is misleading under imbalance — report precision/recall/F1 and a
     confusion matrix; ROC-AUC for ranking.
7. **Interpret** — coefficients (standardized) or feature importances, with the caveat that these are
   associational, not causal. State what actually drives the prediction.
8. **Report** — the metric on held-out data, vs the baseline, plus honest limitations (sample size,
   imbalance, features not available, possible overfitting).

## Rigor rules
- **Never report training-set performance as if it were generalization.** If test >> train-appropriate,
  suspect leakage.
- **Imbalanced classes:** don't quote bare accuracy (99% by predicting the majority class is worthless);
  use F1 / balanced accuracy / AUC and consider resampling or class weights.
- **Small data:** prefer cross-validation; report the spread, not one lucky split.
- Feature importance / coefficients describe the MODEL, not the world (confounding remains).
- If the data can't support a reliable model (too few rows, no signal), **say so** — don't dress up noise.

## Example
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(classification_report(yte, m.predict(Xte)))     # held-out metrics
```
Report: baseline, held-out metric(s), what drives it, and the limitations.
