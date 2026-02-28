# Philippine Food Price Dashboard & Multi-Model Forecasting

Interactive visualization and ML forecasting system for Philippine food commodity prices, using official [WFP Humanitarian Data Exchange](https://data.humdata.org/dataset/wfp-food-prices-for-philippines) data.

## Features

- **Interactive Dashboard** (`dashboard.html`) — Explore historical prices across 73 commodities, 17 regions, and 3 price types (Farm Gate, Retail, Wholesale) from 2000 to present
- **Multi-Model Comparison** (`comparison.html`) — Compare 5 ML models' forecasting accuracy with per-commodity breakdowns, error distributions, and future price projections
- **Daily Auto-Updater** (`daily_update.py`) — Automatically downloads latest WFP data, detects changes, and retrains models
- **Forecasts through Dec 2027** with confidence grading per commodity

## Models

| Model | Description |
|-------|-------------|
| Gradient Boosting | Sequential ensemble of weak learners — best for non-linear trends |
| Random Forest | Bagged decision trees — robust to noise and outliers |
| Extra Trees | Randomized split trees — fast, low-variance predictions |
| Ridge Regression | Regularized linear model — captures stable linear trends |
| KNN (k=10) | Distance-weighted nearest neighbors — captures local patterns |

## Data

- **Source**: WFP VAM Food Prices — Philippines
- **Coverage**: 143K+ records, 2000–present, updated weekly
- **Commodities**: 73 (rice, vegetables, meat, fish, eggs, spices, etc.)
- **Regions**: 17 Philippine administrative regions

## Setup

```bash
pip install scikit-learn pandas numpy
```

### Run the training pipeline
```bash
python retrain_model.py
```

### Serve the dashboards
```bash
python -m http.server 8787
# Open http://localhost:8787/dashboard.html
# Open http://localhost:8787/comparison.html
```

### Schedule daily updates (Windows)
```cmd
schtasks /create /tn "FoodPriceDashboard" /tr "python D:\ML\Website\daily_update.py" /sc daily /st 06:00
```

## File Structure

| File | Purpose |
|------|---------|
| `dashboard.html` | Main historical data exploration dashboard |
| `comparison.html` | Multi-model comparison & forecast dashboard |
| `retrain_model.py` | Multi-model training pipeline (5 models) |
| `build_dashboard.py` | Generates self-contained dashboard.html |
| `daily_update.py` | Auto-updater with hash-based change detection |
| `model_comparison.json` | Pre-computed model results consumed by comparison.html |

## License

Data provided by [WFP](https://www.wfp.org/) under Creative Commons Attribution 4.0.
