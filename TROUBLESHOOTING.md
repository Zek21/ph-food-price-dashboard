# Troubleshooting Guide

This guide helps resolve common issues with the Philippine Food Price Dashboard.

## Quick Diagnostics

Run this command to check if all files are present:

```bash
python3 -c "
import os
from pathlib import Path

required_files = [
    'retrain_model.py',
    'build_dashboard.py',
    'daily_update.py',
    'dashboard.html',
    'comparison.html'
]

print('File Check:')
for f in required_files:
    exists = '✓' if Path(f).exists() else '✗'
    print(f'  {exists} {f}')
"
```

## Common Issues

### 1. Missing Data File Error

**Error:** `FileNotFoundError: Data file not found: wfp_food_prices_phl_latest.csv`

**Solution:**
1. Download the WFP data file:
   ```bash
   curl -L -o wfp_food_prices_phl_latest.csv \
     "https://data.humdata.org/dataset/ea251823-8694-47b4-82d0-7d27f00e8aba/resource/9a842d72-0d7d-4922-ad0e-eb8106c1ab0e/download/wfp_food_prices_phl.csv"
   ```

2. Or run the daily updater:
   ```bash
   python daily_update.py --force
   ```

### 2. Dashboard Shows "Failed to load model_comparison.json"

**Cause:** Browser security prevents loading local files via `file://` protocol.

**Solution:** Serve via HTTP:
```bash
python -m http.server 8080
```
Then open: http://localhost:8080/comparison.html

### 3. Ridge Regression Shows Unrealistic Forecasts

**Fixed in latest version.** If you still see this:

1. Update to the latest code
2. Re-run the model training:
   ```bash
   python retrain_model.py
   ```

The new version caps predictions at 5x the recent average to prevent runaway forecasts.

### 4. Empty Trend Charts

**Cause:** No data for selected commodity/price type combination.

**Solution:**
- Try selecting different commodities
- Check if `model_comparison.json` was generated successfully
- Re-run: `python retrain_model.py`

### 5. Import Errors (Missing Dependencies)

**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install scikit-learn pandas numpy
```

For a complete list:
```bash
pip install scikit-learn>=1.0.0 pandas>=1.3.0 numpy>=1.21.0
```

### 6. Memory Error During Training

**Error:** `MemoryError` or system freezes

**Solution:**
1. Close other applications
2. Reduce the dataset size (filter by date range)
3. Or increase system swap space:
   ```bash
   # Linux
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### 7. Dashboard Not Updating After Retraining

**Solution:**
1. Clear browser cache (Ctrl+F5 or Cmd+Shift+R)
2. Check file timestamps:
   ```bash
   ls -lh model_comparison.json dashboard.html
   ```
3. Verify data was regenerated:
   ```bash
   tail -20 update_log.txt
   ```

## Data Quality Checks

Run these checks to validate your data:

```python
import pandas as pd

# Load data
df = pd.read_csv('wfp_food_prices_phl_latest.csv')

# Check for issues
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Null prices: {df['price'].isna().sum():,}")
print(f"Zero prices: {(df['price'] == 0).sum():,}")
print(f"Negative prices: {(df['price'] < 0).sum():,}")
print(f"Commodities: {df['commodity'].nunique()}")
print(f"Regions: {df['admin1'].nunique()}")

# Show duplicates
duplicates = df.duplicated(subset=['date', 'commodity', 'admin1', 'pricetype']).sum()
print(f"Duplicate rows: {duplicates:,}")
```

## Performance Issues

### Slow Dashboard Loading

1. **Reduce data range** in `retrain_model.py`:
   ```python
   # Filter to recent years only
   df = df[df['year'] >= 2015]
   ```

2. **Aggregate predictions** before visualization

3. **Use dashboard_enhanced.html** which has optimized rendering

### Long Training Time

Expected times on typical hardware:
- Small dataset (<10K rows): 30 seconds
- Medium dataset (10-100K rows): 2-5 minutes
- Large dataset (>100K rows): 10-20 minutes

To speed up:
1. Reduce number of model variants
2. Filter to fewer commodities
3. Use faster models (Extra Trees instead of Gradient Boosting)

## Validation Commands

### Test Python Scripts

```bash
# Syntax check
python3 -m py_compile retrain_model.py build_dashboard.py daily_update.py

# Run with validation
python retrain_model.py 2>&1 | grep -i error
```

### Test Dashboard Locally

```bash
# Start server
python -m http.server 8080 &

# Test URLs
curl -s http://localhost:8080/dashboard.html | head -1
curl -s http://localhost:8080/comparison.html | head -1
curl -s http://localhost:8080/model_comparison.json | python -m json.tool | head -5

# Stop server
killall python
```

## Getting Help

If issues persist:

1. Check the log file: `cat update_log.txt`
2. Review recent changes: `git log --oneline -5`
3. Verify file permissions: `ls -l *.py *.html`
4. Open an issue on GitHub with:
   - Error message
   - Output of diagnostic commands above
   - System information (`uname -a`, `python --version`)

## Advanced Debugging

Enable verbose logging:

```python
# Add to top of retrain_model.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor resource usage:

```bash
# During training, run in another terminal:
watch -n 1 'ps aux | grep python | grep -v grep; free -h'
```

Check JSON validity:

```bash
python -m json.tool model_comparison.json > /dev/null && echo "✓ Valid JSON" || echo "✗ Invalid JSON"
```
