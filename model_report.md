# Philippine Food Price -- Model Comparison Report

## Summary

- **Best Model:** Ridge Regression
- **Best MAPE:** 0.0%
- **Total Models:** 5
- **Commodities:** 2

## Per-Model Metrics

| Model | MAPE | MAE | R2 | Bias | Rank |
|-------|------|-----|------|------|------|
| Ridge Regression | 0.0% | 0.01 | 1.0 | -0.0% | 1 |
| Gradient Boosting | 3.2% | 1.83 | 0.8171 | -3.0% | 2 |
| Extra Trees | 3.9% | 2.25 | 0.7285 | -3.8% | 3 |
| Random Forest | 4.6% | 2.67 | 0.6413 | -4.6% | 4 |
| KNN (k=10) | 5.6% | 3.18 | 0.5699 | -5.6% | 5 |

## Recommendations

**1.** Use Ridge Regression as primary model (MAPE: 0.0%)
   *Lowest MAPE across all commodities on validation set.*
