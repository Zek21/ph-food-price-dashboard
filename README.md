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
- **Coverage**: 100K+ records, 2000–present, updated weekly (exact totals are regenerated into the dashboard JSON)
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

### Serve the dashboards (simple static)
```bash
python -m http.server 8787
# Open http://localhost:8787/dashboard.html
# Open http://localhost:8787/comparison.html
```

### Serve with API (recommended)
```bash
python api_server.py
# Dashboards:  http://localhost:8787/dashboard_enhanced.html
# Data Quality: http://localhost:8787/data_quality.html
# API:          http://localhost:8787/api/health
```

### Schedule daily updates (Windows)
```cmd
schtasks /create /tn "FoodPriceDashboard" /tr "python D:\ML\Website\daily_update.py" /sc daily /st 06:00
```

## File Structure

| File | Purpose |
|------|---------|
| `dashboard.html` | Main historical data exploration dashboard |
| `dashboard_enhanced.html` | Enhanced dashboard with 7 Chart.js visualizations |
| `comparison.html` | Multi-model comparison & forecast dashboard |
| `comparison_enhanced.html` | Enhanced comparison with 8 charts |
| `data_quality.html` | Data quality monitoring dashboard |
| `api_server.py` | Lightweight HTTP API server (stdlib, port 8787) |
| `retrain_model.py` | Multi-model training pipeline (5 models) |
| `build_dashboard.py` | Generates self-contained dashboard.html |
| `daily_update.py` | Auto-updater with hash-based change detection |
| `dashboard_data.json` | Historical price data (117K+ records) |
| `model_comparison.json` | Pre-computed model results consumed by comparison.html |

## API Reference

The API server (`api_server.py`) provides a lightweight REST API for programmatic access to the dashboard data. It uses Python's stdlib `http.server` — no external dependencies required.

### Start the server
```bash
python api_server.py          # Starts on port 8787
python api_server.py --port 9000  # Custom port
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server health check — uptime, data file status |
| `/api/commodities` | GET | List all 73 commodities with category, record count, price range |
| `/api/regions` | GET | List all 17 regions with record counts |
| `/api/forecast?commodity=X` | GET | Forecast data for a commodity (all 5 models) |
| `/api/data-quality` | GET | Data quality metrics — coverage, anomalies, model accuracy |

### Example Requests

```bash
# Health check
curl http://localhost:8787/api/health

# List commodities
curl http://localhost:8787/api/commodities

# Get forecast for Rice
curl "http://localhost:8787/api/forecast?commodity=Rice%20(regular%2C%20milled)"

# Data quality report
curl http://localhost:8787/api/data-quality
```

### Response Examples

**GET /api/health**
```json
{
  "status": "ok",
  "timestamp": "2026-03-18T15:59:48+08:00",
  "data_files": {
    "dashboard_data.json": {"loaded": true, "modified": "2026-03-08T07:00:00+08:00"},
    "model_comparison.json": {"loaded": true, "modified": "2026-03-10T05:35:02+08:00"}
  },
  "server": {"port": 8787, "pid": 12345, "uptime_s": 42.5}
}
```

**GET /api/forecast?commodity=Corn**
```json
{
  "commodity": "Corn",
  "forecasts": {
    "Gradient Boosting": {"2026-02": 25.3, "2026-03": 25.7, ...},
    "Extra Trees": {"2026-02": 24.9, ...}
  },
  "accuracy": {
    "Gradient Boosting": {"mape": 5.2, "mae": 1.3, "r2": 0.94}
  }
}
```

## License

Data provided by [WFP](https://www.wfp.org/) under Creative Commons Attribution 4.0.
