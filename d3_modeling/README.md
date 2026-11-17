# D3 modeling

Compares a persistence baseline with three candidates (Poisson GLM, random forest, Poisson gradient boosting) for next-year NPA counts, using the D2 temporal holdout (development ≤ 2024, test = 2025, 2026 excluded). The D3 report is [`../D3_Report.pdf`](../D3_Report.pdf).

```bash
pip install -r d3_modeling/requirements.txt
python d3_modeling/evaluate_models.py   # ~30 s; writes d3_modeling/results/
```

Inputs are only the two committed aggregates in `dashboard/data/`. Validation: expanding-window rolling origin; tuning folds 2019–2021 (CMPD: 2020–2021), comparison folds 2022–2024, then one refit and one 2025 test score per model. Selection: lowest comparison-fold MAE, with a one-standard-error preference for the simpler model.

| Output (`results/<source>/`) | Contents |
|---|---|
| `cv_fold_metrics.csv`, `cv_summary.csv` | Per-fold and mean metrics for all models (MAE, RMSE, Poisson deviance, bias, Spearman, top-10% overlap) |
| `tuning_log.csv` | Every hyperparameter setting tried and its tuning-fold MAE |
| `test_2025_metrics.csv` | Single held-out 2025 evaluation (+ a level-adjusted diagnostic MAE) |
| `error_analysis.csv` | Selected model vs. baseline by prior-volume quintile, actual change, history length, fold year |
| `test_2025_largest_errors.csv`, `predictions.csv`, `run_info.json` | Row-level predictions and run metadata |

`source` is `violent` (primary: broad Violent Crime count) or `cmpd` (sensitivity: CMPD report count). Figures are in `results/`.
