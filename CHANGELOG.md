# Changelog

All notable changes to the Philippine Food Price Dashboard project.

## [Unreleased] - 2026-02-28

### Fixed

#### Critical Fixes

- **Ridge Regression Explosive Forecasts**: Fixed issue where Ridge Regression model predictions for Corn commodity were growing unrealistically (694.89 PHP by 2027). Added validation to cap predictions at 5x recent average, with fallback to 1.5x increase if exceeded.
  - Location: `retrain_model.py:442-447`
  - Impact: Prevents unrealistic forecasts across all models

#### Data Validation

- **Missing Data File Handling**: Added file existence checks before attempting to load data files
  - `retrain_model.py`: Raises `FileNotFoundError` with helpful message
  - `build_dashboard.py`: Gracefully handles missing predicted data, continues with historical only
  - `daily_update.py`: Validates script paths before execution

- **Data Quality Checks**: Added validation for required columns and minimum data requirements
  - Validates presence of required columns: `date`, `price`, `commodity`, `admin1`, `pricetype`
  - Ensures at least 100 valid records after cleaning
  - Handles date parsing errors gracefully with `errors='coerce'`
  - Location: `retrain_model.py:206-223`

#### Error Handling

- **Enhanced HTML Error Messages**: Improved error display in dashboard HTML files
  - Better formatted error messages with color-coded styling
  - Includes specific instructions for running HTTP server
  - Validates data structure before initialization
  - Adds console logging for debugging
  - Location: `comparison.html:402-427`, `comparison_enhanced.html:647-672`

- **Subprocess Timeout Handling**: Added timeout protection for model retraining
  - 600-second timeout with explicit error on timeout
  - Better error message truncation (last 2000 chars only)
  - Location: `daily_update.py:110-121`

#### Empty Data Series

- **Empty Trend Series**: Fixed potential issue where empty trend series could be added to dashboard
  - Filters out date entries with no data
  - Skips entirely empty series
  - Location: `retrain_model.py:498-500`, `build_dashboard.py:92-93`

### Improved

#### Robustness

- **Prediction Validation**: All model predictions now validated to prevent unrealistic values
  - Maximum cap at 5x recent 12-month average
  - Fallback to 1.5x increase if cap exceeded
  - Applies to all 5 model types

- **Better Logging**: Enhanced logging throughout data pipeline
  - File paths shown in error messages
  - Data quality metrics logged
  - Validation results displayed

#### User Experience

- **Better Error Messages**: All error messages now include:
  - Clear description of the problem
  - Specific steps to resolve
  - Relevant file paths or commands
  - Helpful formatting and color coding

- **HTTP Server Instructions**: Dashboard error pages now show exact commands needed:
  ```bash
  python -m http.server 8080
  ```
  With correct URLs for each dashboard variant

### Documentation

- **TROUBLESHOOTING.md**: Added comprehensive troubleshooting guide covering:
  - Common issues and solutions
  - Data quality checks
  - Performance optimization
  - Validation commands
  - Advanced debugging techniques

### Technical Details

#### Files Modified

1. `retrain_model.py` - 27 lines added
   - Data validation (lines 201-223)
   - Prediction capping (lines 444-447)

2. `build_dashboard.py` - 18 lines added
   - File existence checks (lines 22-23, 53-57)
   - Empty data validation (lines 46-47)

3. `daily_update.py` - 11 lines added
   - Script validation (lines 101-104)
   - Timeout handling (lines 110-121)

4. `comparison.html` - 14 lines modified
   - Enhanced error display (lines 402-427)

5. `comparison_enhanced.html` - 14 lines modified
   - Enhanced error display (lines 647-672)

#### Validation Added

- File existence: 3 locations
- Data structure: 2 locations
- Prediction sanity: 1 location
- Column presence: 1 location
- Minimum data size: 1 location

#### Error Types Now Handled

1. `FileNotFoundError` - Missing data/script files
2. `ValueError` - Invalid/insufficient data
3. `subprocess.TimeoutExpired` - Long-running operations
4. `TypeError` - Malformed JSON data
5. HTTP errors - Network/server issues

### Performance

No performance impact from added validation - all checks are O(1) or O(n) with small constants.

### Breaking Changes

None. All changes are backward compatible.

### Migration Notes

For existing installations:
1. No migration needed - simply pull latest code
2. Re-run `python retrain_model.py` to regenerate forecasts with new validation
3. Existing data files and configurations work unchanged

### Known Issues

None currently identified.

### Future Improvements

Planned for next release:
- [ ] Add data quality metrics dashboard
- [ ] Implement automated anomaly detection
- [ ] Add email notifications for daily_update.py failures
- [ ] Performance profiling and optimization
- [ ] Unit tests for validation functions
- [ ] Integration tests for full pipeline

---

## Previous Releases

See git history for changes before this tracking began.
