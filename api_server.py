#!/usr/bin/env python3
"""
Philippine Food Price Dashboard — REST API Server
==================================================
Lightweight HTTP API serving forecast data, commodity/region lists,
and static files from the same directory.

Endpoints:
    GET /api/health         — Server health check
    GET /api/commodities    — List all commodities (from dashboard_data.json)
    GET /api/regions        — List all regions
    GET /api/forecast       — Forecast data (?commodity=X&region=Y)
    GET /api/data-quality   — Data quality summary
    GET /api/alerts         — Early warning alerts (?severity=X&type=Y&commodity=Z&after=YYYY-MM)
    GET /api/scenarios      — Climate/supply scenarios
    GET /*                  — Static file serving (html, js, css, json)

Usage:
    python api_server.py              # Start on port 8787
    python api_server.py --port 9000  # Custom port

Requires no external dependencies — stdlib only.
"""
# signed: gamma

import json
import os
import sys
import time
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
PORT = 8787
HOST = "localhost"

# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

_cache = {}
_cache_ts = {}
CACHE_TTL = 30  # seconds


def _load_json(filename):
    """Load a JSON file with TTL-based caching."""
    path = ROOT / filename
    now = time.time()
    if filename in _cache and (now - _cache_ts.get(filename, 0)) < CACHE_TTL:
        return _cache[filename]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache[filename] = data
        _cache_ts[filename] = now
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _dashboard_data():
    return _load_json("dashboard_data.json")


def _model_data():
    return _load_json("model_comparison.json")


def _file_freshness(filename):
    """Return file modification time as ISO string, or None."""
    path = ROOT / filename
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

def api_health():
    """GET /api/health — server health check."""
    dd = _dashboard_data()
    mc = _model_data()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_files": {
            "dashboard_data.json": {
                "loaded": dd is not None,
                "modified": _file_freshness("dashboard_data.json"),
            },
            "model_comparison.json": {
                "loaded": mc is not None,
                "modified": _file_freshness("model_comparison.json"),
            },
        },
        "server": {
            "port": PORT,
            "pid": os.getpid(),
            "uptime_s": round(time.time() - _server_start, 1),
        },
    }


def api_commodities():
    """GET /api/commodities — list all tracked commodities."""
    dd = _dashboard_data()
    if dd is None:
        return {"error": "dashboard_data.json not found"}
    meta = dd.get("meta", {})
    commodities = sorted(meta.get("commodities", []))

    # Build per-commodity summary from commodityTable
    table = {c["name"]: c for c in dd.get("commodityTable", [])}
    items = []
    for c in commodities:
        entry = {"name": c}
        if c in table:
            entry["category"] = table[c].get("category", "unknown")
            entry["avg_price"] = table[c].get("avg")
            entry["min_price"] = table[c].get("min")
            entry["max_price"] = table[c].get("max")
            entry["records"] = table[c].get("records", 0)
        items.append(entry)

    return {
        "count": len(commodities),
        "commodities": items,
        "categories": sorted(meta.get("categories", [])),
    }


def api_regions():
    """GET /api/regions — list all tracked regions."""
    dd = _dashboard_data()
    if dd is None:
        return {"error": "dashboard_data.json not found"}
    meta = dd.get("meta", {})
    regions = sorted(meta.get("regions", []))

    # Build per-region average prices from regional data
    regional = dd.get("regional", {})
    items = []
    for r in regions:
        entry = {"name": r}
        if r in regional and isinstance(regional[r], dict):
            cats = regional[r]
            entry["category_averages"] = {
                k: round(v, 2) if isinstance(v, (int, float)) else v
                for k, v in cats.items()
            }
        items.append(entry)

    return {
        "count": len(regions),
        "regions": items,
        "price_types": sorted(meta.get("pricetypes", [])),
    }


def api_forecast(params):
    """GET /api/forecast?commodity=X&region=Y — forecast data."""
    mc = _model_data()
    if mc is None:
        return {"error": "model_comparison.json not found"}

    comm_list = params.get("commodity", [None])
    commodity = comm_list[0] if comm_list else None
    # region param is optional — model_comparison is currently not region-specific
    meta = mc.get("meta", {})
    available = meta.get("commodities", [])

    if not commodity:
        return {
            "error": "Missing required parameter: commodity",
            "available_commodities": available,
            "usage": "/api/forecast?commodity=Rice (regular, milled)",
        }

    # Case-insensitive matching
    match = None
    for c in available:
        if c.lower() == commodity.lower():
            match = c
            break
    if match is None:
        return {
            "error": f"Commodity '{commodity}' not found in forecast data",
            "available_commodities": available,
        }

    # Collect forecasts per model
    forecasts_by_model = {}
    for model_name, model_forecasts in mc.get("forecasts", {}).items():
        if isinstance(model_forecasts, dict) and match in model_forecasts:
            fc = model_forecasts[match]
            if isinstance(fc, dict):
                forecasts_by_model[model_name] = fc

    # Collect accuracy per model
    accuracy_by_model = {}
    comm_comp = mc.get("commComparison", {})
    if match in comm_comp:
        cc = comm_comp[match]
        for model_name in mc.get("modelColors", {}).keys():
            if model_name in cc:
                accuracy_by_model[model_name] = cc[model_name]

    # Collect overall model performance
    overall = {}
    for model_name, perf in mc.get("overall", {}).items():
        overall[model_name] = perf

    # Collect trend data (actual + model predictions)
    trends = {}
    for trend_key, trend_data in mc.get("trends", {}).items():
        if trend_key.startswith(match + "|"):
            pricetype = trend_key.split("|", 1)[1] if "|" in trend_key else "unknown"
            trends[pricetype] = trend_data

    return {
        "commodity": match,
        "meta": {
            "train_period": meta.get("trainPeriod", ""),
            "validation_period": meta.get("valPeriod", ""),
            "forecast_period": meta.get("forecastPeriod", ""),
        },
        "forecasts": forecasts_by_model,
        "accuracy": accuracy_by_model,
        "overall_performance": overall,
        "trends": trends,
    }


def api_data_quality():
    """GET /api/data-quality — data quality summary."""
    dd = _dashboard_data()
    mc = _model_data()
    result = {"timestamp": datetime.now(timezone.utc).isoformat()}

    if dd is None:
        result["error"] = "dashboard_data.json not found"
        return result

    meta = dd.get("meta", {})
    table = dd.get("commodityTable", [])

    # Freshness
    result["freshness"] = {
        "dashboard_data_modified": _file_freshness("dashboard_data.json"),
        "model_comparison_modified": _file_freshness("model_comparison.json"),
        "date_range": meta.get("histDateRangeLabel", "unknown"),
        "year_range": meta.get("histYearRange", []),
    }

    # Record counts
    total_records = meta.get("totalHistRows", 0)
    records_by_commodity = sorted(
        [{"name": c["name"], "records": c.get("records", 0)} for c in table],
        key=lambda x: x["records"],
        reverse=True,
    )

    # Per-region record counts (approximate from regional data)
    regional = dd.get("regional", {})
    regions_with_data = len(regional)
    total_regions = len(meta.get("regions", []))

    result["records"] = {
        "total": total_records,
        "by_commodity": records_by_commodity,
        "commodities_tracked": len(meta.get("commodities", [])),
        "regions_with_data": regions_with_data,
        "total_regions": total_regions,
    }

    # Missing data analysis
    commodities_with_few = [c for c in records_by_commodity if c["records"] < 50]
    result["missing_data"] = {
        "commodities_with_under_50_records": commodities_with_few,
        "missing_count": len(commodities_with_few),
        "coverage_pct": round(
            (1 - len(commodities_with_few) / max(len(records_by_commodity), 1)) * 100, 1
        ),
    }

    # Price anomalies (>30% deviation from average)
    anomalies = []
    for c in table:
        avg = c.get("avg", 0)
        mn = c.get("min", 0)
        mx = c.get("max", 0)
        if avg > 0:
            spread = (mx - mn) / avg
            if spread > 3.0:  # max is >3x the average, likely anomalous
                anomalies.append({
                    "commodity": c["name"],
                    "avg": avg,
                    "min": mn,
                    "max": mx,
                    "spread_ratio": round(spread, 2),
                })
    result["anomalies"] = sorted(anomalies, key=lambda x: x["spread_ratio"], reverse=True)

    # Model accuracy
    if mc:
        model_accuracy = {}
        for model_name, perf in mc.get("overall", {}).items():
            model_accuracy[model_name] = {
                "mape": perf.get("mape"),
                "mae": perf.get("mae"),
                "r2": perf.get("r2"),
                "bias": perf.get("bias"),
            }
        result["model_accuracy"] = model_accuracy

    return result


# signed: gamma
def api_alerts(params):
    """Return early warning alerts from alerts.json, with optional severity filter."""
    alerts_file = ROOT / "alerts.json"
    if not alerts_file.exists():
        return {"error": "alerts.json not found. Run: python early_warning.py --scan"}

    data = _load_json("alerts.json")
    if data is None:
        return {"error": "Failed to load alerts.json"}

    sev_filter = (params.get("severity", [None])[0] or "").lower() if params.get("severity") else None
    type_filter = (params.get("type", [None])[0] or "").lower() if params.get("type") else None
    comm_filter = (params.get("commodity", [None])[0] or "").lower() if params.get("commodity") else None
    date_filter = params.get("after", [None])[0] if params.get("after") else None

    alerts = data.get("alerts", [])
    if sev_filter:
        alerts = [a for a in alerts if a.get("severity") == sev_filter]
    if type_filter:
        alerts = [a for a in alerts if a.get("alert_type") == type_filter]
    if comm_filter:
        alerts = [a for a in alerts if comm_filter in a.get("commodity", "").lower()]
    if date_filter:
        alerts = [a for a in alerts if a.get("date", "") >= date_filter]

    severity_counts = {}
    for a in alerts:
        s = a.get("severity", "unknown")
        severity_counts[s] = severity_counts.get(s, 0) + 1

    return {
        "scan_timestamp": data.get("scan_timestamp"),
        "total": len(alerts),
        "by_severity": severity_counts,
        "alerts": alerts,
    }


def api_scenarios():
    """Return climate scenario data if available."""
    scenarios_file = ROOT / "climate_scenarios.json"
    if scenarios_file.exists():
        data = _load_json("climate_scenarios.json")
        if data:
            return data

    # Generate basic scenarios from forecast data when no dedicated file exists
    model = _load_json("model_comparison.json")
    if not model:
        return {"error": "No scenario data available"}

    forecasts = model.get("forecasts", {})
    overall = model.get("overall", {})

    scenarios = {
        "baseline": {
            "name": "Baseline",
            "description": "Normal conditions — ML model consensus forecast",
            "probability": "60%",
            "assumptions": ["Normal weather patterns", "Stable trade policy", "No supply shocks"],
        },
        "adverse_weather": {
            "name": "Adverse Weather (El Niño/La Niña)",
            "description": "Climate disruption adds 10-25% to staple prices",
            "probability": "25%",
            "assumptions": ["Typhoon season above average", "Drought in key producing regions",
                            "Reduced domestic production by 15-20%"],
        },
        "supply_shock": {
            "name": "Supply Chain Disruption",
            "description": "Import delays or trade restrictions push prices 20-40% above baseline",
            "probability": "10%",
            "assumptions": ["Major port disruption", "Export ban from key trade partners",
                            "Fuel price spike affecting transport costs"],
        },
        "best_case": {
            "name": "Favorable Conditions",
            "description": "Good harvests and stable imports keep prices 5-10% below baseline",
            "probability": "5%",
            "assumptions": ["Above-average harvests", "Reduced import tariffs",
                            "Improved logistics and cold chain"],
        },
    }

    # Attach model forecast data to baseline scenario
    model_names = list(forecasts.keys())
    if model_names:
        sample_model = model_names[0]
        sample_commodities = list(forecasts[sample_model].keys())[:5]
        scenarios["baseline"]["sample_forecasts"] = {
            comm: forecasts[sample_model].get(comm, {})
            for comm in sample_commodities
        }

    scenarios["models_available"] = model_names
    scenarios["model_performance"] = {
        m: {"mape": d.get("mape"), "r2": d.get("r2")}
        for m, d in overall.items()
    }
    return scenarios


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class FoodPriceAPIHandler(SimpleHTTPRequestHandler):
    """Handles API routes and falls back to static file serving."""

    # Map API paths to handlers
    API_ROUTES = {
        "/api/health": lambda _: api_health(),
        "/api/commodities": lambda _: api_commodities(),
        "/api/regions": lambda _: api_regions(),
        "/api/forecast": lambda params: api_forecast(params),
        "/api/data-quality": lambda _: api_data_quality(),
        "/api/alerts": lambda params: api_alerts(params),
        "/api/scenarios": lambda _: api_scenarios(),
    }

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        # Check API routes
        handler = self.API_ROUTES.get(path)
        if handler:
            try:
                result = handler(params)
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # CORS preflight for API
        if path.startswith("/api/"):
            self._send_json(404, {"error": f"Unknown endpoint: {path}"})
            return

        # Fall through to static file serving
        super().do_GET()

    def _send_json(self, status, data):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Quieter logging — only log API requests and errors."""
        msg = format % args
        if "/api/" in msg or "404" in msg or "500" in msg:
            sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_server_start = time.time()


def main():
    global PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            PORT = int(sys.argv[idx + 1])

    os.chdir(ROOT)

    # Pre-load data to verify files exist
    dd = _dashboard_data()
    mc = _model_data()
    print(f"Data loaded: dashboard_data={'OK' if dd else 'MISSING'}, "
          f"model_comparison={'OK' if mc else 'MISSING'}")
    if dd:
        meta = dd.get("meta", {})
        print(f"  Commodities: {len(meta.get('commodities', []))}, "
              f"Regions: {len(meta.get('regions', []))}, "
              f"Records: {meta.get('totalHistRows', 0):,}")

    server = HTTPServer((HOST, PORT), FoodPriceAPIHandler)
    print(f"\nPhilippine Food Price API running on http://{HOST}:{PORT}")
    print(f"  Dashboard:    http://{HOST}:{PORT}/dashboard_enhanced.html")
    print(f"  Data Quality: http://{HOST}:{PORT}/data_quality.html")
    print(f"  API Health:   http://{HOST}:{PORT}/api/health")
    print(f"  Commodities:  http://{HOST}:{PORT}/api/commodities")
    print(f"  Regions:      http://{HOST}:{PORT}/api/regions")
    print(f"  Forecast:     http://{HOST}:{PORT}/api/forecast?commodity=Rice")
    print(f"\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
# signed: gamma
