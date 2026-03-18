"""Validate generated dashboard JSON outputs for consistency.

This script is intentionally lightweight so it can run after every rebuild.
It focuses on the counts and cross-file metadata that drive dashboard copy,
helping prevent stale counters or mismatched generated assets from slipping in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_JSON = BASE_DIR / "dashboard_data.json"
COMPARISON_JSON = BASE_DIR / "model_comparison.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    expect(DASHBOARD_JSON.exists(), f"Missing file: {DASHBOARD_JSON.name}", errors)
    expect(COMPARISON_JSON.exists(), f"Missing file: {COMPARISON_JSON.name}", errors)
    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return 1

    dashboard = load_json(DASHBOARD_JSON)
    comparison = load_json(COMPARISON_JSON)

    d_meta = dashboard.get("meta", {})
    c_meta = comparison.get("meta", {})

    dashboard_commodities = set(d_meta.get("commodities", []))
    comparison_commodities = set(c_meta.get("commodities", []))
    dashboard_regions = set(d_meta.get("regions", []))
    comparison_regions = set(c_meta.get("regions", []))
    dashboard_price_types = set(d_meta.get("pricetypes", []))
    comparison_price_types = set(c_meta.get("pricetypes", []))
    dashboard_categories = set(d_meta.get("categories", []))
    category_keys = set((dashboard.get("categories") or {}).keys())

    expect(bool(dashboard_commodities), "dashboard_data.json meta.commodities is empty", errors)
    expect(bool(comparison_commodities), "model_comparison.json meta.commodities is empty", errors)
    expect(dashboard_commodities == comparison_commodities, "Commodity sets differ between dashboard_data.json and model_comparison.json", errors)
    expect(dashboard_regions == comparison_regions, "Region sets differ between dashboard_data.json and model_comparison.json", errors)
    expect(dashboard_price_types == comparison_price_types, "Price-type sets differ between dashboard_data.json and model_comparison.json", errors)
    expect(dashboard_categories == category_keys, "dashboard_data.json meta.categories does not match category summary keys", errors)

    expect(
        d_meta.get("totalHistRows") == c_meta.get("totalActualRows"),
        "Historical row count in dashboard_data.json does not match totalActualRows in model_comparison.json",
        errors,
    )

    dashboard_trends = dashboard.get("trends", {})
    comparison_trends = comparison.get("trends", {})
    expect(set(dashboard_trends.keys()) == set(comparison_trends.keys()), "Trend series keys differ between dashboard_data.json and model_comparison.json", errors)

    commodity_table = dashboard.get("commodityTable", [])
    expect(bool(commodity_table), "dashboard_data.json commodityTable is empty", errors)
    expect(len(commodity_table) <= len(dashboard_commodities), "commodityTable has more rows than the full commodity list", errors)

    for row in commodity_table:
        name = row.get("name")
        category = row.get("category")
        expect(name in dashboard_commodities, f"Commodity table row references unknown commodity: {name}", errors)
        expect(category in dashboard_categories, f"Commodity table row references unknown category: {category}", errors)

    map_points = dashboard.get("mapPoints", [])
    for point in map_points:
        expect(point.get("lat") not in (None, 0), f"Map point has invalid latitude: {point.get('name')}", errors)
        expect(point.get("lng") not in (None, 0), f"Map point has invalid longitude: {point.get('name')}", errors)
        expect(point.get("region") in dashboard_regions, f"Map point references unknown region: {point.get('name')}", errors)

    if errors:
        print("Validation failed. The dashboard counters/data are out of sync:")
        for err in errors:
            print(f"[FAIL] {err}")
        return 1

    print(
        "Validation passed: "
        f"{len(dashboard_commodities)} commodities, "
        f"{len(dashboard_regions)} regions, "
        f"{len(dashboard_price_types)} price types, "
        f"{d_meta.get('totalHistRows', 0):,} historical rows."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())