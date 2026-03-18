# Philippine Food Price Dashboard & Multi-Model Forecasting

Interactive visualization, ML forecasting, and early warning system for Philippine food commodity prices, using official [WFP Humanitarian Data Exchange](https://data.humdata.org/dataset/wfp-food-prices-for-philippines) data.

<!-- signed: gamma -->

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                        DATA INGESTION                               │
 │  daily_update.py ──► WFP CSV ──► exogenous_features.py              │
 │                       (117K+       (ENSO ONI, USD/PHP,              │
 │                        records)     FAO Food Price Index)            │
 └──────────────┬────────────────────────────┬─────────────────────────┘
                │                            │
                ▼                            ▼
 ┌──────────────────────────┐  ┌─────────────────────────────────────┐
 │   TRAINING PIPELINE      │  │   CLIMATE & EXOGENOUS ANALYSIS      │
 │                          │  │                                     │
 │  retrain_model.py        │  │  climate_scenarios.py               │
 │    ├─ Gradient Boosting  │  │    └─ ENSO impact analysis          │
 │    ├─ Random Forest      │  │       (7 states × 3 lag periods)    │
 │    ├─ Extra Trees        │  │                                     │
 │    ├─ Ridge Regression   │  │  exogenous_features.py              │
 │    └─ KNN                │  │    ├─ NOAA ONI Index                │
 │                          │  │    ├─ ECB USD/PHP rates             │
 │  lstm_model.py           │  │    └─ FAO Food Price Index          │
 │    └─ 2-layer LSTM       │  │                                     │
 │                          │  └─────────────────────────────────────┘
 │  ensemble_model.py       │
 │    └─ Stacking (GB+ET+RF │               │
 │       → Ridge meta)      │               ▼
 └──────────┬───────────────┘  ┌─────────────────────────────────────┐
            │                  │   EARLY WARNING SYSTEM               │
            ▼                  │                                     │
 ┌──────────────────────────┐  │  early_warning.py                   │
 │   OUTPUT DATA            │  │    ├─ Spike detection (>15%)        │
 │                          │  │    ├─ Year-over-year alerts (>20%)  │
 │  dashboard_data.json     │  │    ├─ Regional divergence (z>1.5)   │
 │  model_comparison.json   │  │    └─ Model divergence (>15%)       │
 │  lstm_predictions.json   │  │                                     │
 │  climate_scenarios.json  │  │  alerts.json (949 alerts)           │
 │  alerts.json             │  └─────────────────────────────────────┘
 └──────────┬───────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                     PRESENTATION LAYER                              │
 │                                                                     │
 │  api_server.py (port 8787, stdlib)     Dashboards:                  │
 │    /api/health                          dashboard_enhanced.html     │
 │    /api/commodities                     comparison_enhanced.html    │
 │    /api/regions                         data_quality.html           │
 │    /api/forecast                        early_warning.html          │
 │    /api/data-quality                                                │
 │    /api/alerts                         Chart.js + Leaflet.js maps   │
 │    /api/scenarios                      Dark/light theme support     │
 └──────────────────────────────────────────────────────────────────────┘
```

## Features

- **Interactive Dashboards** — Explore historical prices across 73 commodities, 17 regions, and 3 price types with Chart.js visualizations and Leaflet.js maps
- **7 ML Models** — 5 scikit-learn models + LSTM neural network + stacking ensemble
- **Early Warning System** — FEWS NET-inspired anomaly detection with 4-level severity classification
- **Climate Integration** — ENSO impact analysis with 3-6 month lag, exchange rate and FAO index tracking
- **REST API** — 7-endpoint API server using Python stdlib (zero dependencies)
- **Daily Auto-Updater** — Hash-based change detection and automatic model retraining
- **Forecasts through Dec 2027** with confidence grading per commodity

## Models

| Model | Type | Description | Dependencies |
|-------|------|-------------|-------------|
| Gradient Boosting | Scikit-learn | Sequential ensemble of weak learners — best for non-linear trends | scikit-learn |
| Random Forest | Scikit-learn | Bagged decision trees — robust to noise and outliers | scikit-learn |
| Extra Trees | Scikit-learn | Randomized split trees — fast, low-variance predictions | scikit-learn |
| Ridge Regression | Scikit-learn | Regularized linear model — captures stable linear trends | scikit-learn |
| KNN (k=10) | Scikit-learn | Distance-weighted nearest neighbors — captures local patterns | scikit-learn |
| **LSTM** | PyTorch | 2-layer LSTM (128 hidden) with 12-month sliding windows | torch |
| **Stacking Ensemble** | Scikit-learn | GB + ExtraTrees + RandomForest → Ridge meta-learner | scikit-learn |

## Data

- **Source**: [WFP VAM Food Prices — Philippines](https://data.humdata.org/dataset/wfp-food-prices-for-philippines)
- **Coverage**: 117K+ records, 2000–present, updated weekly
- **Commodities**: 73 (rice, vegetables, meat, fish, eggs, spices, etc.)
- **Regions**: 17 Philippine administrative regions
- **Exogenous**: ENSO ONI Index (NOAA), USD/PHP exchange rate (ECB), FAO Food Price Index

## Installation

### Core dependencies (scikit-learn models)
```bash
pip install -r requirements.txt
# Installs: numpy, pandas, scikit-learn, joblib
```

### LSTM model (optional)
```bash
pip install torch
```

### Full installation
```bash
pip install -r requirements.txt torch
```

> **Note**: The API server and early warning system use Python stdlib only — no extra packages needed.

## Quick Start

### 1. Train baseline models
```bash
python retrain_model.py
```

### 2. Train LSTM model (optional, requires PyTorch)
```bash
python lstm_model.py --train --predict --epochs 100
```

### 3. Train stacking ensemble (optional)
```bash
python ensemble_model.py --train --evaluate
```

### 4. Fetch exogenous data & run climate analysis
```bash
python exogenous_features.py --fetch --save
python climate_scenarios.py --analyze
```

### 5. Generate early warning alerts
```bash
python early_warning.py --scan --output alerts.json
```

### 6. Start the API server
```bash
python api_server.py
# Dashboards:    http://localhost:8787/dashboard_enhanced.html
# Comparison:    http://localhost:8787/comparison_enhanced.html
# Data Quality:  http://localhost:8787/data_quality.html
# Early Warning: http://localhost:8787/early_warning.html
# API:           http://localhost:8787/api/health
```

### Schedule daily updates (Windows)
```cmd
schtasks /create /tn "FoodPriceDashboard" /tr "python D:\ML\Website\daily_update.py" /sc daily /st 06:00
```

---

## LSTM Model (`lstm_model.py`)

Per-commodity LSTM neural network for time-series price forecasting.

### Architecture

```
Input (seq_len=12, features=6)
  → LSTM(input=6, hidden=128, layers=2, dropout=0.2, batch_first=True)
  → FC(128, 64) → ReLU → Dropout(0.2)
  → FC(64, 1)
```

**Features per timestep**: normalized price, month_sin, month_cos, year_normalized, region_encoded, pricetype_encoded

**Training**: AdamW optimizer, Huber loss (robust to outliers), ReduceLROnPlateau scheduler, gradient clipping (max_norm=1.0), early stopping (patience=10). Validation holdout: last 24 months.

### Usage

```bash
# Train all commodities
python lstm_model.py --train --epochs 100

# Train and generate forecasts
python lstm_model.py --train --predict --epochs 50 --lr 0.001

# Predict for a specific commodity
python lstm_model.py --predict --commodity "Rice (Regular Milled)" --region "National Capital region"
```

**Output**: `lstm_predictions.json` — dashboard-compatible format with per-commodity forecasts, validation metrics (MAPE, MAE, RMSE, R², bias), and architecture metadata.

**Trained models**: saved to `.lstm_models/` as PyTorch checkpoints with embedded scaler parameters for reproducible inference.

---

## Ensemble Model (`ensemble_model.py`)

Stacking ensemble that combines three tree-based models via a Ridge regression meta-learner.

### Architecture

| Component | Model | Trees | Max Depth | Config |
|-----------|-------|-------|-----------|--------|
| Base | Gradient Boosting | 200 | 4 | lr=0.05, subsample=0.8 |
| Base | Extra Trees | 200 | 10 | min_samples_leaf=5 |
| Base | Random Forest | 200 | 10 | min_samples_leaf=3 |
| **Meta** | **Ridge** | — | — | **alpha=1.0** |

**Cross-validation**: KFold(n_splits=5, shuffle=False) preserves temporal ordering.

**Features**: 12 engineered features — price lags (1, 3, 6, 12 months), moving averages (3, 6, 12 months), price diffs (1, 12 months), temporal (year, month_sin, month_cos) + categorical encodings.

### Usage

```bash
python ensemble_model.py --train --evaluate
python ensemble_model.py --train --evaluate --cv 3
```

---

## Exogenous Features (`exogenous_features.py`)

Fetches and merges external data sources that influence Philippine food prices.

| Source | API | Data | Update Frequency |
|--------|-----|------|------------------|
| ENSO ONI Index | [NOAA CPC](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | El Niño / La Niña / Neutral classification | Monthly |
| USD/PHP Exchange Rate | [Frankfurter (ECB)](https://api.frankfurter.app/) | Monthly averaged daily rates | Daily → monthly |
| FAO Food Price Index | Embedded historical data (base 2014-2016=100) | Global food price benchmark | Monthly |

### Usage

```bash
python exogenous_features.py --fetch --save       # Download all sources and save
python exogenous_features.py --status             # Show cached data status
```

**Output**: `exogenous_data.json` — year-month indexed DataFrame with oni_value, enso_state, usd_php_rate, fao_fpi.

---

## Climate Scenarios (`climate_scenarios.py`)

Analyzes historical ENSO-price relationships and generates climate impact scenarios.

### ENSO States Analyzed

| State | ONI Range | Example Impact |
|-------|-----------|----------------|
| Strong El Niño | ONI > 1.5 | Drought → rice/corn price spikes |
| Moderate El Niño | 1.0 – 1.5 | Moderate agricultural stress |
| Weak El Niño | 0.5 – 1.0 | Minor yield reduction |
| Neutral | -0.5 – 0.5 | Baseline conditions |
| Weak La Niña | -1.0 – -0.5 | Flooding risk, slight price pressure |
| Moderate La Niña | -1.5 – -1.0 | Significant flood/typhoon risk |
| Strong La Niña | ONI < -1.5 | Severe weather disruption |

**Impact lag analysis**: 0, 3, and 6 months (ENSO affects Philippine food prices with 3-6 month delay).

### Usage

```bash
python climate_scenarios.py --analyze                    # Full analysis (all scenarios)
python climate_scenarios.py --scenario "strong_el_nino"  # Specific scenario
python climate_scenarios.py --summary                    # Quick summary table
```

**Output**: `climate_scenarios.json` — per-commodity, per-region impact statistics for each ENSO state.

---

## Early Warning System (`early_warning.py` + `early_warning.html`)

FEWS NET-inspired food price anomaly detection system with 4-level severity classification.

### Alert Types

| Type | Detection Method | Thresholds |
|------|-----------------|------------|
| **Spike** | Predicted price vs 3-month rolling average | >15% low, >20% medium, >30% high, >50% critical |
| **Year-over-Year** | Same month vs prior year comparison | >20% low, >30% medium, >50% high, >80% critical |
| **Regional Divergence** | Z-score from national average price | >1.5σ low, >2.0σ medium, >2.5σ high, >3.0σ critical |
| **Model Divergence** | Inter-model forecast spread | >15% low, >20% medium, >30% high, >50% critical |

### Policy Recommendations

Each alert includes actionable recommendations based on severity:
- **Low**: Monitor and prepare contingency plans
- **Medium**: Activate buffer stock review, increase import surveillance
- **High**: Release strategic reserves, coordinate with regional offices
- **Critical**: Invoke price ceiling authority (RA 7581), emergency import authorization

### Usage

```bash
# Full scan across all commodities and regions
python early_warning.py --scan --output alerts.json

# Filter by commodity or region
python early_warning.py --commodity "Rice (regular, milled)" --region "ARMM"

# Filter by severity
python early_warning.py --severity critical
```

### Dashboard

`early_warning.html` provides an interactive alert dashboard with:
- Leaflet.js map showing regional alert distribution
- Chart.js severity breakdown and timeline
- Filterable alert table with expandable policy recommendations
- Dark/light theme matching other dashboards

**Output**: `alerts.json` — structured alerts with severity, type, commodity, region, prices, thresholds, and recommendations.

---

## API Reference

The API server (`api_server.py`) provides a lightweight REST API using Python's stdlib `http.server` — **zero external dependencies**. JSON responses are cached with a 30-second TTL.

### Start the server

```bash
python api_server.py              # Start on port 8787
python api_server.py --port 9000  # Custom port
```

### Endpoints

| Endpoint | Method | Parameters | Description |
|----------|--------|------------|-------------|
| `/api/health` | GET | — | Server health check — uptime, data file status |
| `/api/commodities` | GET | — | List all 73 commodities with category, record count, price range |
| `/api/regions` | GET | — | List all 17 regions with record counts |
| `/api/forecast?commodity=X` | GET | `commodity` (required) | Forecast data for a commodity (all models) |
| `/api/data-quality` | GET | — | Data quality metrics — coverage, anomalies, model accuracy |
| `/api/alerts` | GET | `severity`, `type`, `commodity`, `after` | Early warning alerts with optional filters |
| `/api/scenarios` | GET | — | Climate/supply scenarios from ENSO analysis |

### Example Requests

```bash
# Health check
curl http://localhost:8787/api/health

# List commodities
curl http://localhost:8787/api/commodities

# Get forecast for Rice
curl "http://localhost:8787/api/forecast?commodity=Rice%20(regular%2C%20milled)"

# Critical alerts only
curl "http://localhost:8787/api/alerts?severity=critical"

# Climate scenarios
curl http://localhost:8787/api/scenarios

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

**GET /api/alerts?severity=critical**
```json
{
  "scan_timestamp": "2026-03-18T10:00:00",
  "filters": {"severity": "critical"},
  "summary": {"total": 103, "critical": 103},
  "alerts": [
    {
      "commodity": "Shrimp (endeavor)",
      "region": "Region V",
      "type": "yoy_increase",
      "severity": "critical",
      "current_price": 420.0,
      "predicted_price": 810.0,
      "threshold": 80,
      "recommendation": "Invoke price ceiling authority per RA 7581..."
    }
  ]
}
```

---

## Test Suite

Tests are in the `tests/` directory and use pytest.

| Test File | Coverage |
|-----------|----------|
| `test_retrain_model.py` | Baseline model training pipeline |
| `test_daily_update.py` | Daily data refresh and hash-based change detection |
| `test_data_quality.py` | Data validation and anomaly detection |
| `test_ensemble.py` | Ensemble model training and stacking evaluation |

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_ensemble.py -v

# Run with coverage (if pytest-cov installed)
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## File Structure

### Scripts

| File | Purpose | Dependencies |
|------|---------|-------------|
| `retrain_model.py` | Train 5 scikit-learn models | scikit-learn, pandas, numpy |
| `lstm_model.py` | Train/predict with LSTM neural network | torch, scikit-learn, pandas, numpy |
| `ensemble_model.py` | Stacking ensemble (GB+ET+RF → Ridge) | scikit-learn, pandas, numpy, joblib |
| `exogenous_features.py` | Fetch ENSO, exchange rate, FAO data | pandas, numpy (+ urllib for APIs) |
| `climate_scenarios.py` | ENSO impact analysis and scenarios | pandas, numpy |
| `early_warning.py` | Anomaly detection and alert generation | stdlib only |
| `api_server.py` | REST API server (7 endpoints) | stdlib only |
| `build_dashboard.py` | Generate self-contained dashboard HTML | pandas |
| `daily_update.py` | Auto-updater with change detection | pandas |
| `validate_dashboard_outputs.py` | Validate generated JSON outputs | stdlib only |

### Dashboards

| File | Purpose |
|------|---------|
| `dashboard_enhanced.html` | Enhanced historical data explorer (7 Chart.js visualizations, Leaflet map) |
| `comparison_enhanced.html` | Multi-model comparison with 8 charts |
| `data_quality.html` | Data quality monitoring — freshness, coverage, anomalies |
| `early_warning.html` | Early warning alert dashboard — map, timeline, severity filters |

### Data Files

| File | Generated By | Consumed By | Description |
|------|-------------|-------------|-------------|
| `dashboard_data.json` | `build_dashboard.py` | API, dashboards | 117K+ price records (1.4 MB) |
| `model_comparison.json` | `retrain_model.py` | API, dashboards | 5-model forecasts and accuracy metrics |
| `lstm_predictions.json` | `lstm_model.py` | Comparison dashboard | LSTM forecasts with architecture metadata |
| `climate_scenarios.json` | `climate_scenarios.py` | API, dashboards | ENSO impact by commodity and region |
| `alerts.json` | `early_warning.py` | API, early warning dashboard | Active price anomaly alerts |
| `exogenous_data.json` | `exogenous_features.py` | `climate_scenarios.py` | ENSO ONI, USD/PHP, FAO FPI |

---

## License

Data provided by [WFP](https://www.wfp.org/) under Creative Commons Attribution 4.0.
