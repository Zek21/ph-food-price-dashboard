## Dashboard audit protocol

- For `dashboard.html` and `comparison.html`, always load the pages in a browser over HTTP so JSON assets resolve correctly.
- During audits, always check rendered UI, source code, and generated JSON/data together.
- Never trust reported model metrics until lag, rolling, and difference features are confirmed to be target-leak free.
- Validation in this repository starts at `2024-01-01` unless code and UI copy are intentionally updated together.
- When page copy mentions counts or date ranges, prefer values derived from JSON metadata over hardcoded text.
- Keep temporary audit artifacts out of the workspace when possible, and remove stale screenshots/log remnants after verification.
- For dashboard reviews, always check console/runtime issues, missing assets, broken controls, and content/data mismatches before considering the task complete.