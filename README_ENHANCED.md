# Philippine Food Price Dashboard & Multi-Model Forecasting — Enhanced Edition

Interactive visualization and ML forecasting system for Philippine food commodity prices, using official [WFP Humanitarian Data Exchange](https://data.humdata.org/dataset/wfp-food-prices-for-philippines) data.

## 🆕 What's New in Enhanced Edition

### Interactive Features
- **🔍 Zoom & Pan**: Scroll to zoom, drag to pan on all charts
- **📥 Data Export**: Export charts as PNG, tables as CSV, full data as JSON
- **⚡ Real-time Search**: Instant commodity filtering across all dashboards
- **🎯 Click-to-Filter**: Interactive model selection and highlighting
- **⌨️ Keyboard Shortcuts**: Ctrl+E (export), Ctrl+R (reset), Esc (clear)
- **💾 State Persistence**: Remembers your selections using localStorage
- **✨ Smooth Animations**: Professional fade-in and slide-up transitions

### Enhanced Dashboards

#### `comparison_enhanced.html` (74KB)
Multi-model comparison dashboard with:
- 8 interactive charts with export buttons
- Global action bar with quick access controls
- Enhanced tooltips with rich information
- Model performance radar chart
- Accuracy grade distribution
- Per-commodity MAPE comparison
- Zoom hints and contextual help

#### `dashboard_enhanced.html` (31KB)
Historical price dashboard with:
- Interactive map with Leaflet
- Global search and filtering
- Export capabilities for all visualizations
- Enhanced price trend analysis
- Regional comparisons
- Year-over-year analysis
- Top movers identification

## Features

### Data Visualization
- **Interactive Dashboard** (`dashboard_enhanced.html`) — Explore historical prices across 73 commodities, 17 regions, and 3 price types (Farm Gate, Retail, Wholesale) from 2000 to present
- **Multi-Model Comparison** (`comparison_enhanced.html`) — Compare 5 ML models' forecasting accuracy with per-commodity breakdowns, error distributions, and future price projections
- **Daily Auto-Updater** (`daily_update.py`) — Automatically downloads latest WFP data, detects changes, and retrains models
- **Forecasts through Dec 2027** with confidence grading per commodity

### Machine Learning Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Gradient Boosting** | Sequential ensemble of weak learners | Best for non-linear trends |
| **Random Forest** | Bagged decision trees | Robust to noise and outliers |
| **Extra Trees** | Randomized split trees | Fast, low-variance predictions |
| **Ridge Regression** | Regularized linear model | Captures stable linear trends |
| **KNN (k=10)** | Distance-weighted nearest neighbors | Captures local patterns |

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

### Serve the enhanced dashboards
```bash
python -m http.server 8787
# Open http://localhost:8787/dashboard_enhanced.html
# Open http://localhost:8787/comparison_enhanced.html
```

### Schedule daily updates

**Windows (Task Scheduler):**
```cmd
schtasks /create /tn "FoodPriceDashboard" /tr "python daily_update.py" /sc daily /st 06:00
```

**Linux/macOS (cron):**
```bash
# Run at 06:00 daily — add via: crontab -e
0 6 * * * cd /path/to/ph-food-price-dashboard && python daily_update.py
```

## File Structure

| File | Purpose |
|------|---------|
| `dashboard_enhanced.html` | **NEW** Enhanced historical data exploration dashboard |
| `comparison_enhanced.html` | **NEW** Enhanced multi-model comparison & forecast dashboard |
| `dashboard.html` | Original historical data dashboard |
| `comparison.html` | Original multi-model comparison dashboard |
| `retrain_model.py` | Multi-model training pipeline (5 models) |
| `build_dashboard.py` | Generates self-contained dashboard.html |
| `daily_update.py` | Auto-updater with hash-based change detection |
| `model_comparison.json` | Pre-computed model results consumed by comparison.html |
| `ENHANCEMENTS.md` | **NEW** Detailed documentation of all enhancements |
| `test_enhancements.html` | **NEW** Automated test suite for enhanced features |

## Technical Stack

### Core Libraries
- **Chart.js 4.4.1** — Powerful, flexible charting library
- **chartjs-plugin-zoom 2.0.1** — Zoom and pan functionality
- **Hammer.js 2.0.8** — Touch gesture support
- **Leaflet 1.9.4** — Interactive map visualization
- **scikit-learn** — Machine learning models
- **pandas** — Data manipulation
- **numpy** — Numerical computing

### Architecture
- Pure vanilla JavaScript (no framework dependencies)
- Self-contained HTML files (can work offline)
- LocalStorage for state persistence
- Responsive CSS Grid layout
- Dark mode optimized design

## Usage Guide

### Keyboard Shortcuts
- `Ctrl+E` / `Cmd+E`: Export current view
- `Ctrl+R` / `Cmd+R`: Reset all filters
- `Esc`: Clear selection

### Chart Interactions
- **Zoom In**: Scroll up on chart
- **Zoom Out**: Scroll down on chart
- **Pan**: Click and drag chart
- **Reset View**: Double-click chart
- **Details**: Hover over data points

### Exporting Data
1. **Chart Images**: Click "📥 PNG" button on any chart
2. **Table Data**: Click "📊 CSV" button on tables
3. **Full Dataset**: Click "📥 Export All Data" in action bar

### Filtering & Search
- Use global search box to filter commodities
- Click model tiles to highlight that model
- Click score cards to select a model
- Selections are saved automatically

## Performance

- Page load: <1 second (with cached resources)
- Render time: ~750ms animation duration
- Smooth 60fps animations
- Optimized for datasets up to 10,000+ points

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome/Edge | ✅ Full | Recommended |
| Firefox | ✅ Full | All features work |
| Safari | ✅ Full | iOS gestures supported |
| Mobile | ✅ Full | Touch gestures enabled |

## Testing

Run the automated test suite:
```bash
python -m http.server 8787
# Open http://localhost:8787/test_enhancements.html
```

The test suite validates:
- File existence
- Feature implementation
- Library integration
- Export functionality
- Search and filtering

## Contributing

Contributions are welcome! Areas for enhancement:
1. WebGL rendering for larger datasets
2. Real-time collaborative filtering
3. Custom theme builder
4. Advanced statistical overlays
5. Automated insight generation

## License

Data provided by [WFP](https://www.wfp.org/) under Creative Commons Attribution 4.0.

## Acknowledgments

- World Food Programme for providing the data
- Chart.js team for the excellent visualization library
- scikit-learn developers for ML tools

---

**Enhanced Edition** — February 2026 — Version 2.0
