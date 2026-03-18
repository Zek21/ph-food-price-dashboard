#!/usr/bin/env python3
"""
Philippine Food Price Early Warning System
==========================================
Inspired by FEWS NET methodology: multi-factor analysis with severity classification.
Detects emerging price anomalies 1-3 months before retail peaks using ML forecasts.

Usage:
    python early_warning.py --scan --output alerts.json
    python early_warning.py --commodity "Rice (regular, milled)"
    python early_warning.py --commodity "Corn" --region "National Capital Region"
    python early_warning.py --severity critical
"""
# signed: gamma

import json
import argparse
import statistics
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DASHBOARD_DATA = BASE_DIR / "dashboard_data.json"
MODEL_COMPARISON = BASE_DIR / "model_comparison.json"
ALERTS_OUTPUT = BASE_DIR / "alerts.json"

# --- Severity thresholds (FEWS NET inspired 4-level classification) ---
THRESHOLDS = {
    "spike_pct": {            # predicted vs 3-month rolling avg
        "low": 0.15,          # 15-20%
        "medium": 0.20,       # 20-30%
        "high": 0.30,         # 30-50%
        "critical": 0.50,     # >50%
    },
    "yoy_pct": {              # year-over-year increase
        "low": 0.20,          # 20-30%
        "medium": 0.30,       # 30-50%
        "high": 0.50,         # 50-80%
        "critical": 0.80,     # >80%
    },
    "regional_zscore": {      # deviation from national average (z-scores)
        "low": 1.5,
        "medium": 2.0,
        "high": 2.5,
        "critical": 3.0,
    },
    "model_divergence_pct": { # disagreement between ML models
        "low": 0.15,
        "medium": 0.25,
        "high": 0.40,
        "critical": 0.60,
    },
}

POLICY_RECOMMENDATIONS = {
    "spike": {
        "low":      "Monitor weekly. Increase market price surveillance frequency.",
        "medium":   "Pre-position buffer stock releases. Alert regional price councils.",
        "high":     "Activate import facilitation. Coordinate with NFA for buffer stock release. "
                    "Consider temporary tariff reduction on affected commodities.",
        "critical": "EMERGENCY: Immediate buffer stock release. Fast-track import permits. "
                    "Deploy transport subsidies to affected regions. Activate social protection "
                    "programs (conditional cash transfers, food vouchers).",
    },
    "yoy_increase": {
        "low":      "Track inflation trend. Review supply chain for bottlenecks.",
        "medium":   "Engage traders' associations. Investigate supply-side constraints. "
                    "Consider forward contracting with producers.",
        "high":     "Activate price stabilization fund. Negotiate bilateral import agreements. "
                    "Deploy mobile markets in price-stressed areas.",
        "critical": "EMERGENCY: Invoke price ceiling authority per RA 7581. Coordinate with DTI "
                    "for price freeze on basic necessities. Immediate import augmentation.",
    },
    "regional_divergence": {
        "low":      "Monitor transport costs and local supply conditions.",
        "medium":   "Deploy transport subsidies to isolated regions. Improve market connectivity.",
        "high":     "Activate inter-regional commodity transfers. Emergency logistics deployment. "
                    "Investigate possible hoarding or market manipulation.",
        "critical": "EMERGENCY: Direct government procurement and distribution to affected region. "
                    "Deploy NFA rolling stores. Coordinate with DSWD for food packs.",
    },
    "model_divergence": {
        "low":      "Continue monitoring. Models show normal uncertainty range.",
        "medium":   "Increase data collection frequency. Cross-validate with market reports.",
        "high":     "High forecast uncertainty. Prepare contingency plans for both scenarios. "
                    "Engage domain experts for qualitative assessment.",
        "critical": "Forecast reliability compromised. Fall back to expert judgment and real-time "
                    "market monitoring. Do not rely solely on model predictions.",
    },
}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_severity(value: float, thresholds: dict) -> str:
    """Classify severity based on threshold dict (ascending order)."""
    if value >= thresholds["critical"]:
        return "critical"
    if value >= thresholds["high"]:
        return "high"
    if value >= thresholds["medium"]:
        return "medium"
    if value >= thresholds["low"]:
        return "low"
    return "none"


def rolling_average(prices: list[float], window: int = 3) -> float | None:
    """Compute rolling average of last `window` values."""
    if len(prices) < window:
        return None
    return statistics.mean(prices[-window:])


def detect_spike_alerts(trends: dict, forecasts: dict) -> list[dict]:
    """
    Detect price spikes: when forecast exceeds 3-month rolling average by >15%.
    Uses FEWS NET-inspired multi-threshold severity classification.
    """
    alerts = []
    for model_name, model_forecasts in forecasts.items():
        for commodity, date_prices in model_forecasts.items():
            # Build historical price series from trends
            trend_key = None
            for tk in trends:
                if tk.startswith(commodity):
                    trend_key = tk
                    break
            if not trend_key:
                continue

            hist_data = trends[trend_key]
            sorted_dates = sorted(hist_data.keys())
            actual_prices = []
            for d in sorted_dates:
                val = hist_data[d].get("actual")
                if val is not None:
                    actual_prices.append((d, val))

            if len(actual_prices) < 3:
                continue

            recent_vals = [p for _, p in actual_prices[-6:]]
            roll_avg = rolling_average(recent_vals)
            if roll_avg is None or roll_avg <= 0:
                continue

            forecast_dates = sorted(date_prices.keys())
            for fc_date in forecast_dates:
                fc_price = date_prices[fc_date]
                if fc_price is None or fc_price <= 0:
                    continue

                pct_change = (fc_price - roll_avg) / roll_avg
                severity = classify_severity(pct_change, THRESHOLDS["spike_pct"])
                if severity == "none":
                    continue

                alerts.append({
                    "alert_type": "spike",
                    "commodity": commodity,
                    "region": "National",
                    "date": fc_date,
                    "severity": severity,
                    "current_price": round(roll_avg, 2),
                    "predicted_price": round(fc_price, 2),
                    "pct_change": round(pct_change * 100, 1),
                    "threshold": f">{int(THRESHOLDS['spike_pct'][severity]*100)}%",
                    "model": model_name,
                    "recommendation": POLICY_RECOMMENDATIONS["spike"][severity],
                })
    return alerts


def detect_yoy_alerts(trends: dict) -> list[dict]:
    """
    Seasonal adjustment: flag when same-month price exceeds prior year by >20%.
    """
    alerts = []
    for trend_key, hist_data in trends.items():
        commodity = trend_key.split("|")[0] if "|" in trend_key else trend_key
        sorted_dates = sorted(hist_data.keys())

        date_map = {}
        for d in sorted_dates:
            val = hist_data[d].get("actual")
            if val is not None:
                date_map[d] = val

        for date_str, price in date_map.items():
            try:
                year, month = date_str.split("-")
                prev_year_key = f"{int(year)-1}-{month}"
            except ValueError:
                continue

            prev_price = date_map.get(prev_year_key)
            if prev_price is None or prev_price <= 0:
                continue

            yoy_change = (price - prev_price) / prev_price
            severity = classify_severity(yoy_change, THRESHOLDS["yoy_pct"])
            if severity == "none":
                continue

            alerts.append({
                "alert_type": "yoy_increase",
                "commodity": commodity,
                "region": "National",
                "date": date_str,
                "severity": severity,
                "current_price": round(price, 2),
                "predicted_price": round(prev_price, 2),
                "pct_change": round(yoy_change * 100, 1),
                "threshold": f">{int(THRESHOLDS['yoy_pct'][severity]*100)}% YoY",
                "model": "Historical",
                "recommendation": POLICY_RECOMMENDATIONS["yoy_increase"][severity],
            })
    return alerts


def detect_regional_alerts(regional: dict) -> list[dict]:
    """
    Flag regions where commodity category prices diverge significantly from national average.
    Uses z-score based detection.
    """
    alerts = []
    categories = set()
    for region_data in regional.values():
        categories.update(region_data.keys())

    for category in categories:
        prices_by_region = {}
        for region, data in regional.items():
            val = data.get(category)
            if val is not None and val > 0:
                prices_by_region[region] = val

        if len(prices_by_region) < 3:
            continue

        values = list(prices_by_region.values())
        mean_price = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        if stdev <= 0:
            continue

        for region, price in prices_by_region.items():
            zscore = abs(price - mean_price) / stdev
            severity = classify_severity(zscore, THRESHOLDS["regional_zscore"])
            if severity == "none":
                continue

            direction = "above" if price > mean_price else "below"
            alerts.append({
                "alert_type": "regional_divergence",
                "commodity": category,
                "region": region,
                "date": datetime.now().strftime("%Y-%m"),
                "severity": severity,
                "current_price": round(price, 2),
                "predicted_price": round(mean_price, 2),
                "pct_change": round((price - mean_price) / mean_price * 100, 1),
                "threshold": f"z-score >{THRESHOLDS['regional_zscore'][severity]} ({direction})",
                "model": "Regional Analysis",
                "recommendation": POLICY_RECOMMENDATIONS["regional_divergence"][severity],
            })
    return alerts


def detect_model_divergence_alerts(forecasts: dict) -> list[dict]:
    """
    Flag commodities where ML models disagree significantly, indicating forecast uncertainty.
    """
    alerts = []
    commodity_dates = defaultdict(lambda: defaultdict(list))
    for model_name, model_fc in forecasts.items():
        for commodity, date_prices in model_fc.items():
            for fc_date, price in date_prices.items():
                if price is not None and price > 0:
                    commodity_dates[commodity][fc_date].append(
                        (model_name, price)
                    )

    for commodity, dates in commodity_dates.items():
        for fc_date, model_prices in dates.items():
            if len(model_prices) < 3:
                continue
            prices = [p for _, p in model_prices]
            mean_p = statistics.mean(prices)
            if mean_p <= 0:
                continue

            spread = (max(prices) - min(prices)) / mean_p
            severity = classify_severity(spread, THRESHOLDS["model_divergence_pct"])
            if severity == "none":
                continue

            model_details = {m: round(p, 2) for m, p in model_prices}
            alerts.append({
                "alert_type": "model_divergence",
                "commodity": commodity,
                "region": "National",
                "date": fc_date,
                "severity": severity,
                "current_price": round(mean_p, 2),
                "predicted_price": None,
                "pct_change": round(spread * 100, 1),
                "threshold": f"spread >{int(THRESHOLDS['model_divergence_pct'][severity]*100)}%",
                "model": "Multi-Model",
                "recommendation": POLICY_RECOMMENDATIONS["model_divergence"][severity],
                "model_predictions": model_details,
            })
    return alerts


def deduplicate_alerts(alerts: list[dict], keep_highest: bool = True) -> list[dict]:
    """
    Deduplicate alerts: keep only the highest severity per commodity+region+date+type.
    """
    key_map: dict[str, dict] = {}
    for alert in alerts:
        key = f"{alert['alert_type']}|{alert['commodity']}|{alert['region']}|{alert['date']}"
        existing = key_map.get(key)
        if existing is None:
            key_map[key] = alert
        elif keep_highest:
            if SEVERITY_ORDER.index(alert["severity"]) > SEVERITY_ORDER.index(existing["severity"]):
                key_map[key] = alert
    return list(key_map.values())


def run_scan(commodity_filter: str | None = None,
             region_filter: str | None = None,
             severity_filter: str | None = None) -> dict:
    """Run full early warning scan and return structured alert report."""
    dashboard = load_json(DASHBOARD_DATA)
    model_data = load_json(MODEL_COMPARISON)

    trends = dashboard.get("trends", {})
    regional = dashboard.get("regional", {})
    forecasts = model_data.get("forecasts", {})

    all_alerts = []
    all_alerts.extend(detect_spike_alerts(trends, forecasts))
    all_alerts.extend(detect_yoy_alerts(trends))
    all_alerts.extend(detect_regional_alerts(regional))
    all_alerts.extend(detect_model_divergence_alerts(forecasts))

    all_alerts = deduplicate_alerts(all_alerts)

    # Apply filters
    if commodity_filter:
        cf_lower = commodity_filter.lower()
        all_alerts = [a for a in all_alerts if cf_lower in a["commodity"].lower()]
    if region_filter:
        rf_lower = region_filter.lower()
        all_alerts = [a for a in all_alerts if rf_lower in a["region"].lower()]
    if severity_filter:
        all_alerts = [a for a in all_alerts if a["severity"] == severity_filter.lower()]

    # Sort: critical first, then high, then by date
    all_alerts.sort(
        key=lambda a: (-SEVERITY_ORDER.index(a["severity"]), a["date"]),
    )

    # Summary statistics
    severity_counts = {s: 0 for s in SEVERITY_ORDER}
    type_counts = defaultdict(int)
    commodity_counts = defaultdict(int)
    for a in all_alerts:
        severity_counts[a["severity"]] += 1
        type_counts[a["alert_type"]] += 1
        commodity_counts[a["commodity"]] += 1

    top_commodities = sorted(commodity_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "scan_timestamp": datetime.now().isoformat(),
        "filters": {
            "commodity": commodity_filter,
            "region": region_filter,
            "severity": severity_filter,
        },
        "summary": {
            "total_alerts": len(all_alerts),
            "by_severity": severity_counts,
            "by_type": dict(type_counts),
            "top_commodities": [{"name": c, "alerts": n} for c, n in top_commodities],
        },
        "alerts": all_alerts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Philippine Food Price Early Warning System"
    )
    parser.add_argument("--scan", action="store_true", help="Run full scan")
    parser.add_argument("--commodity", type=str, help="Filter by commodity name")
    parser.add_argument("--region", type=str, help="Filter by region")
    parser.add_argument("--severity", type=str, choices=SEVERITY_ORDER,
                        help="Filter by severity level")
    parser.add_argument("--output", type=str, default=str(ALERTS_OUTPUT),
                        help="Output file path (default: alerts.json)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (no summary)")
    args = parser.parse_args()

    if not args.scan and not args.commodity and not args.region and not args.severity:
        args.scan = True

    result = run_scan(
        commodity_filter=args.commodity,
        region_filter=args.region,
        severity_filter=args.severity,
    )

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        s = result["summary"]
        print(f"\n{'='*60}")
        print(f"  PHILIPPINE FOOD PRICE EARLY WARNING SYSTEM")
        print(f"  Scan: {result['scan_timestamp']}")
        print(f"{'='*60}")
        print(f"\n  Total Alerts: {s['total_alerts']}")
        print(f"  ├─ Critical: {s['by_severity']['critical']}")
        print(f"  ├─ High:     {s['by_severity']['high']}")
        print(f"  ├─ Medium:   {s['by_severity']['medium']}")
        print(f"  └─ Low:      {s['by_severity']['low']}")
        if s["by_type"]:
            print(f"\n  By Type:")
            for t, c in s["by_type"].items():
                print(f"    {t}: {c}")
        if s["top_commodities"]:
            print(f"\n  Top Affected Commodities:")
            for item in s["top_commodities"][:5]:
                print(f"    {item['name']}: {item['alerts']} alerts")

        # Show top critical/high alerts
        top_alerts = [a for a in result["alerts"] if a["severity"] in ("critical", "high")][:10]
        if top_alerts:
            print(f"\n  {'─'*56}")
            print(f"  Top Critical/High Alerts:")
            for a in top_alerts:
                sev = a["severity"].upper()
                print(f"\n  [{sev}] {a['commodity']} — {a['region']} ({a['date']})")
                print(f"    Type: {a['alert_type']} | Change: {a['pct_change']}%")
                if a.get("current_price"):
                    print(f"    Price: ₱{a['current_price']} → ₱{a.get('predicted_price', '?')}")
                print(f"    → {a['recommendation'][:100]}...")

        print(f"\n  Alerts saved to: {output_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
