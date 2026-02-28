"""
Build an interactive food price dashboard from WFP historical + predicted data.
Aggregates large datasets and generates a self-contained HTML file.
"""

import csv
import json
import os
from collections import defaultdict

# ─── Configurable paths ─────────────────────────────────────
HIST_DATA_PATH = os.environ.get("HIST_DATA_PATH", "Dataset_WFP2.csv")
PRED_DATA_PATH = os.environ.get("PRED_DATA_PATH", "merged_file_with_location_2.csv")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "dashboard.html")

print("=" * 60)
print("  WFP Philippines Food Price Dashboard Builder")
print("=" * 60)

# ─── 1. Load Historical Data ────────────────────────────────
print("\n[1/4] Loading historical data...")
hist_rows = []
with open(HIST_DATA_PATH, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            price = float(r["Price"])
        except (ValueError, KeyError):
            continue
        hist_rows.append({
            "year": int(r["Year"]),
            "month": int(r["Month"]),
            "region": r["Region"],
            "province": r["Province"],
            "locality": r["Locality"],
            "category": r["Category"],
            "commodity": r["Commodity"],
            "pricetype": r["Pricetype"],
            "price": price,
            "location": r["Location"],
        })
print(f"   Loaded {len(hist_rows):,} historical rows")

# ─── 2. Load Predicted Data (aggregated) ────────────────────
print("\n[2/4] Loading predicted data (aggregating 22M+ rows)...")
pred_agg = defaultdict(lambda: {"sum": 0, "count": 0})
pred_count = 0
with open(PRED_DATA_PATH, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        pred_count += 1
        if pred_count % 5_000_000 == 0:
            print(f"   ...processed {pred_count/1e6:.0f}M rows")
        try:
            price = float(r["Predicted_Price"])
        except (ValueError, KeyError):
            continue
        key = (int(r["Year"]), int(r["Month"]), r["Region"], r["Category"], r["Commodity"], r["Pricetype"])
        pred_agg[key]["sum"] += price
        pred_agg[key]["count"] += 1

print(f"   Processed {pred_count:,} predicted rows → {len(pred_agg):,} aggregated groups")

# ─── 3. Build Dashboard Data ────────────────────────────────
print("\n[3/4] Building dashboard data structures...")

# -- Monthly price trends (by commodity+pricetype, averaged across regions) --
hist_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for r in hist_rows:
    key = f"{r['commodity']}|{r['pricetype']}"
    date_key = f"{r['year']}-{r['month']:02d}"
    hist_trends[key][date_key]["sum"] += r["price"]
    hist_trends[key][date_key]["count"] += 1

pred_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for (year, month, region, cat, comm, pt), v in pred_agg.items():
    key = f"{comm}|{pt}"
    date_key = f"{year}-{month:02d}"
    pred_trends[key][date_key]["sum"] += v["sum"]
    pred_trends[key][date_key]["count"] += v["count"]

# Build combined trends JSON
trends_data = {}
all_keys = set(hist_trends.keys()) | set(pred_trends.keys())
for key in all_keys:
    series = {}
    for dk, v in hist_trends.get(key, {}).items():
        series[dk] = {"price": round(v["sum"] / v["count"], 4), "type": "historical"}
    for dk, v in pred_trends.get(key, {}).items():
        if dk not in series:
            series[dk] = {"price": round(v["sum"] / v["count"], 4), "type": "predicted"}
    if series:
        trends_data[key] = series

# -- Regional averages (latest year of historical data = 2023) --
regional_hist = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for r in hist_rows:
    if r["year"] >= 2020:
        key = r["region"]
        cat = r["category"]
        regional_hist[key][cat]["sum"] += r["price"]
        regional_hist[key][cat]["count"] += 1

regional_data = {}
for region, cats in regional_hist.items():
    regional_data[region] = {}
    for cat, v in cats.items():
        regional_data[region][cat] = round(v["sum"] / v["count"], 4)

# -- Category summary --
cat_summary = defaultdict(lambda: {"sum": 0, "count": 0, "commodities": set()})
for r in hist_rows:
    cat_summary[r["category"]]["sum"] += r["price"]
    cat_summary[r["category"]]["count"] += 1
    cat_summary[r["category"]]["commodities"].add(r["commodity"])

category_data = {}
for cat, v in cat_summary.items():
    category_data[cat] = {
        "avg_price": round(v["sum"] / v["count"], 4),
        "records": v["count"],
        "commodities": len(v["commodities"]),
    }

# -- Commodity list with stats --
comm_stats = defaultdict(lambda: {"prices": [], "category": "", "pricetypes": set()})
for r in hist_rows:
    if r["year"] >= 2022:
        comm_stats[r["commodity"]]["prices"].append(r["price"])
        comm_stats[r["commodity"]]["category"] = r["category"]
        comm_stats[r["commodity"]]["pricetypes"].add(r["pricetype"])

commodity_table = []
for comm, v in sorted(comm_stats.items()):
    prices = v["prices"]
    if prices:
        commodity_table.append({
            "name": comm,
            "category": v["category"],
            "avg": round(sum(prices) / len(prices), 2),
            "min": round(min(prices), 2),
            "max": round(max(prices), 2),
            "records": len(prices),
        })

# -- Location data for map (aggregate by locality) --
loc_data = defaultdict(lambda: {"sum": 0, "count": 0, "lat": 0, "lng": 0, "region": "", "province": ""})
for r in hist_rows:
    if r["year"] >= 2020 and r["location"]:
        loc = r["location"]
        loc_data[r["locality"]]["sum"] += r["price"]
        loc_data[r["locality"]]["count"] += 1
        try:
            parts = loc.split(",")
            loc_data[r["locality"]]["lat"] = float(parts[0])
            loc_data[r["locality"]]["lng"] = float(parts[1])
        except (ValueError, IndexError):
            pass
        loc_data[r["locality"]]["region"] = r["region"]
        loc_data[r["locality"]]["province"] = r["province"]

map_points = []
for locality, v in loc_data.items():
    if v["lat"] != 0 and v["lng"] != 0:
        map_points.append({
            "name": locality,
            "lat": v["lat"],
            "lng": v["lng"],
            "avg_price": round(v["sum"] / v["count"], 2),
            "records": v["count"],
            "region": v["region"],
            "province": v["province"],
        })

# -- Year-over-year inflation by category --
yoy_data = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for r in hist_rows:
    yoy_data[r["year"]][r["category"]]["sum"] += r["price"]
    yoy_data[r["year"]][r["category"]]["count"] += 1
# Add predicted
for (year, month, region, cat, comm, pt), v in pred_agg.items():
    yoy_data[year][cat]["sum"] += v["sum"]
    yoy_data[year][cat]["count"] += v["count"]

yoy_json = {}
for year in sorted(yoy_data.keys()):
    yoy_json[str(year)] = {}
    for cat, v in yoy_data[year].items():
        yoy_json[str(year)][cat] = round(v["sum"] / v["count"], 4)

# -- Commodities & categories & regions lists --
all_commodities = sorted(set(r["commodity"] for r in hist_rows))
all_categories = sorted(set(r["category"] for r in hist_rows))
all_regions = sorted(set(r["region"] for r in hist_rows))
all_pricetypes = sorted(set(r["pricetype"] for r in hist_rows))

# -- Top movers: biggest price changes 2020 vs 2023 --
price_by_year_comm = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for r in hist_rows:
    if r["year"] in (2020, 2023) and r["pricetype"] == "Retail":
        price_by_year_comm[r["commodity"]][r["year"]]["sum"] += r["price"]
        price_by_year_comm[r["commodity"]][r["year"]]["count"] += 1

movers = []
for comm, years in price_by_year_comm.items():
    if 2020 in years and 2023 in years and years[2020]["count"] > 5 and years[2023]["count"] > 5:
        p2020 = years[2020]["sum"] / years[2020]["count"]
        p2023 = years[2023]["sum"] / years[2023]["count"]
        if p2020 > 0:
            change_pct = round((p2023 - p2020) / p2020 * 100, 1)
            movers.append({"commodity": comm, "price_2020": round(p2020, 2), "price_2023": round(p2023, 2), "change_pct": change_pct})
movers.sort(key=lambda x: x["change_pct"], reverse=True)

dashboard_data = {
    "trends": trends_data,
    "regional": regional_data,
    "categories": category_data,
    "commodityTable": commodity_table,
    "mapPoints": map_points,
    "yoy": yoy_json,
    "movers": movers,
    "meta": {
        "commodities": all_commodities,
        "categories": all_categories,
        "regions": all_regions,
        "pricetypes": all_pricetypes,
        "histYearRange": [2000, 2023],
        "predYearRange": [2023, 2027],
        "totalHistRows": len(hist_rows),
        "totalPredRows": pred_count,
    },
}

# ─── 4. Generate HTML Dashboard ─────────────────────────────
print("\n[4/4] Generating HTML dashboard...")

data_json = json.dumps(dashboard_data, separators=(",", ":"))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Philippines Food Price Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e4e6eb;
    --text2: #8b8fa3;
    --accent: #6366f1;
    --accent2: #22d3ee;
    --green: #22c55e;
    --red: #ef4444;
    --orange: #f59e0b;
    --pink: #ec4899;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    padding: 2rem 2rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }}
  .header p {{ color: var(--text2); font-size: 0.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}

  /* Summary Cards */
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .summary-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
  }}
  .summary-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }}
  .summary-card:nth-child(1)::before {{ background: var(--accent); }}
  .summary-card:nth-child(2)::before {{ background: var(--accent2); }}
  .summary-card:nth-child(3)::before {{ background: var(--green); }}
  .summary-card:nth-child(4)::before {{ background: var(--orange); }}
  .summary-card:nth-child(5)::before {{ background: var(--pink); }}
  .summary-card:nth-child(6)::before {{ background: var(--red); }}
  .summary-card .label {{ font-size: 0.75rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary-card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }}
  .summary-card .sub {{ font-size: 0.8rem; color: var(--text2); margin-top: 0.2rem; }}

  /* Chart Cards */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .full-width {{ margin-bottom: 1.5rem; }}
  @media (max-width: 900px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
  }}
  .card .subtitle {{
    font-size: 0.8rem;
    color: var(--text2);
    margin-bottom: 1rem;
  }}
  .card canvas {{ max-height: 350px; }}

  /* Controls */
  .controls {{
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }}
  select, button {{
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    cursor: pointer;
    transition: border-color 0.2s;
  }}
  select:hover, button:hover {{ border-color: var(--accent); }}
  select:focus, button:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(99,102,241,0.2); }}

  /* Table */
  .table-container {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 0.7rem 1rem; border-bottom: 2px solid var(--border); color: var(--text2); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; cursor: pointer; }}
  th:hover {{ color: var(--accent); }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(99,102,241,0.05); }}

  .badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-up {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge-down {{ background: rgba(239,68,68,0.15); color: var(--red); }}

  /* Map */
  #map {{ height: 400px; border-radius: 8px; }}

  /* Movers */
  .mover-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
  }}
  .mover-item:last-child {{ border: none; }}
  .mover-name {{ font-weight: 500; font-size: 0.85rem; }}
  .mover-prices {{ font-size: 0.8rem; color: var(--text2); }}

  .legend-row {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }}
  .tab-group {{
    display: flex;
    gap: 0.3rem;
    margin-bottom: 1rem;
  }}
  .tab-btn {{
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.8rem;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text2);
    cursor: pointer;
  }}
  .tab-btn.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Philippines Food Price Intelligence</h1>
  <p>Historical WFP data (2000-2023) + Random Forest predictions (2023-2027) &mdash; {len(hist_rows):,} historical records &bull; {pred_count:,} predicted records &bull; 73 commodities &bull; 17 regions</p>
</div>

<div class="container">
  <!-- Summary Cards -->
  <div class="summary-grid" id="summaryCards"></div>

  <!-- Price Trend -->
  <div class="card full-width">
    <h2>Price Trends</h2>
    <p class="subtitle">Historical prices + ML predictions. Select a commodity and price type to explore.</p>
    <div class="controls">
      <select id="trendCommodity"></select>
      <select id="trendPriceType"></select>
    </div>
    <div class="legend-row">
      <div class="legend-item"><div class="legend-dot" style="background:#6366f1"></div> Historical</div>
      <div class="legend-item"><div class="legend-dot" style="background:#22d3ee"></div> Predicted (RF Model)</div>
    </div>
    <canvas id="trendChart"></canvas>
  </div>

  <!-- Row: Category + YoY -->
  <div class="grid-2">
    <div class="card">
      <h2>Price by Category</h2>
      <p class="subtitle">Average price per food category (all years)</p>
      <canvas id="categoryChart"></canvas>
    </div>
    <div class="card">
      <h2>Price Trends by Category (Year-over-Year)</h2>
      <p class="subtitle">Annual average price per category, historical + predicted</p>
      <canvas id="yoyChart"></canvas>
    </div>
  </div>

  <!-- Row: Regional + Map -->
  <div class="grid-2">
    <div class="card">
      <h2>Regional Price Comparison</h2>
      <p class="subtitle">Average retail prices by region (2020-2023)</p>
      <div class="controls">
        <select id="regionalCategory">
          <option value="_all">All Categories</option>
        </select>
      </div>
      <canvas id="regionalChart"></canvas>
    </div>
    <div class="card">
      <h2>Price Map</h2>
      <p class="subtitle">Geographic distribution of average prices (2020-2023). Larger = higher price.</p>
      <div id="map"></div>
    </div>
  </div>

  <!-- Row: Movers + Top/Bottom -->
  <div class="grid-2">
    <div class="card">
      <h2>Biggest Price Increases (2020 vs 2023)</h2>
      <p class="subtitle">Retail commodities with the largest % jump</p>
      <div id="moversUp"></div>
    </div>
    <div class="card">
      <h2>Biggest Price Decreases (2020 vs 2023)</h2>
      <p class="subtitle">Retail commodities with the largest % drop</p>
      <div id="moversDown"></div>
    </div>
  </div>

  <!-- Commodity Table -->
  <div class="card full-width">
    <h2>Commodity Reference Table</h2>
    <p class="subtitle">Recent data (2022-2023). Click column headers to sort.</p>
    <div class="controls">
      <select id="tableCategory">
        <option value="_all">All Categories</option>
      </select>
      <input type="text" id="tableSearch" placeholder="Search commodity..." style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;flex:1;min-width:200px;">
    </div>
    <div class="table-container">
      <table id="commTable">
        <thead>
          <tr>
            <th data-col="name">Commodity</th>
            <th data-col="category">Category</th>
            <th data-col="avg">Avg Price (PHP/kg)</th>
            <th data-col="min">Min</th>
            <th data-col="max">Max</th>
            <th data-col="records">Records</th>
          </tr>
        </thead>
        <tbody id="commTableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const D = {data_json};

// ─── Summary Cards ──────────────────────────────────────
(function() {{
  const m = D.meta;
  const cards = [
    {{ label: "Commodities Tracked", value: m.commodities.length, sub: "Across " + m.categories.length + " food categories" }},
    {{ label: "Regions Covered", value: m.regions.length, sub: "Provinces & localities nationwide" }},
    {{ label: "Historical Data", value: m.histYearRange[0] + "-" + m.histYearRange[1], sub: m.totalHistRows.toLocaleString() + " price records" }},
    {{ label: "ML Predictions", value: m.predYearRange[0] + "-" + m.predYearRange[1], sub: m.totalPredRows.toLocaleString() + " forecasted prices" }},
    {{ label: "Price Types", value: m.pricetypes.length, sub: m.pricetypes.join(", ") }},
    {{ label: "Food Categories", value: m.categories.length, sub: Object.values(D.categories).reduce((a,b) => a + b.commodities, 0) + " unique items" }},
  ];
  const el = document.getElementById("summaryCards");
  el.innerHTML = cards.map(c => `
    <div class="summary-card">
      <div class="label">${{c.label}}</div>
      <div class="value">${{c.value}}</div>
      <div class="sub">${{c.sub}}</div>
    </div>
  `).join("");
}})();

// ─── Chart.js Defaults ──────────────────────────────────
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "#2a2d3a";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ─── Price Trend Chart ──────────────────────────────────
(function() {{
  const selComm = document.getElementById("trendCommodity");
  const selPT = document.getElementById("trendPriceType");

  D.meta.commodities.forEach(c => {{
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    if (c === "Rice (regular, milled)") opt.selected = true;
    selComm.appendChild(opt);
  }});
  D.meta.pricetypes.forEach(p => {{
    const opt = document.createElement("option");
    opt.value = p; opt.textContent = p;
    if (p === "Retail") opt.selected = true;
    selPT.appendChild(opt);
  }});

  const ctx = document.getElementById("trendChart").getContext("2d");
  let chart = null;

  function update() {{
    const key = selComm.value + "|" + selPT.value;
    const series = D.trends[key] || {{}};
    const dates = Object.keys(series).sort();

    const histDates = [], histPrices = [], predDates = [], predPrices = [];
    dates.forEach(d => {{
      if (series[d].type === "historical") {{
        histDates.push(d + "-15");
        histPrices.push(series[d].price);
      }} else {{
        predDates.push(d + "-15");
        predPrices.push(series[d].price);
      }}
    }});

    // Add bridge point
    if (histDates.length && predDates.length) {{
      predDates.unshift(histDates[histDates.length - 1]);
      predPrices.unshift(histPrices[histPrices.length - 1]);
    }}

    if (chart) chart.destroy();
    chart = new Chart(ctx, {{
      type: "line",
      data: {{
        datasets: [
          {{
            label: "Historical Price",
            data: histDates.map((d, i) => ({{ x: d, y: histPrices[i] }})),
            borderColor: "#6366f1",
            backgroundColor: "rgba(99,102,241,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHitRadius: 8,
            borderWidth: 2,
          }},
          {{
            label: "Predicted Price (RF)",
            data: predDates.map((d, i) => ({{ x: d, y: predPrices[i] }})),
            borderColor: "#22d3ee",
            backgroundColor: "rgba(34,211,238,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHitRadius: 8,
            borderWidth: 2,
            borderDash: [6, 3],
          }},
        ],
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: "index", intersect: false }},
        scales: {{
          x: {{ type: "time", time: {{ unit: "year" }}, title: {{ display: true, text: "Date" }} }},
          y: {{ title: {{ display: true, text: "Price (PHP/kg)" }}, beginAtZero: false }},
        }},
        plugins: {{ legend: {{ display: false }} }},
      }},
    }});
  }}

  selComm.addEventListener("change", update);
  selPT.addEventListener("change", update);
  update();
}})();

// ─── Category Donut ─────────────────────────────────────
(function() {{
  const labels = Object.keys(D.categories);
  const values = labels.map(k => D.categories[k].avg_price);
  const colors = ["#6366f1","#22d3ee","#22c55e","#f59e0b","#ec4899","#ef4444"];

  new Chart(document.getElementById("categoryChart"), {{
    type: "doughnut",
    data: {{
      labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{{
        data: values,
        backgroundColor: colors,
        borderColor: "#1a1d27",
        borderWidth: 3,
      }}],
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ position: "right", labels: {{ padding: 12, usePointStyle: true, pointStyle: "circle" }} }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: PHP ${{ctx.parsed.toFixed(2)}}/kg avg` }} }},
      }},
    }},
  }});
}})();

// ─── Year-over-Year Chart ───────────────────────────────
(function() {{
  const years = Object.keys(D.yoy).sort();
  const categories = D.meta.categories;
  const colors = ["#6366f1","#22d3ee","#22c55e","#f59e0b","#ec4899","#ef4444"];

  const datasets = categories.map((cat, i) => ({{
    label: cat.charAt(0).toUpperCase() + cat.slice(1),
    data: years.map(y => D.yoy[y][cat] || null),
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length] + "22",
    tension: 0.3,
    pointRadius: 1,
    borderWidth: 2,
  }}));

  new Chart(document.getElementById("yoyChart"), {{
    type: "line",
    data: {{ labels: years, datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: "index", intersect: false }},
      scales: {{
        y: {{ title: {{ display: true, text: "Avg Price (PHP/kg)" }} }},
      }},
      plugins: {{
        legend: {{ position: "bottom", labels: {{ padding: 10, usePointStyle: true, pointStyle: "circle", font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: PHP ${{ctx.parsed.y?.toFixed(2)}}/kg` }} }},
      }},
    }},
  }});
}})();

// ─── Regional Bar Chart ─────────────────────────────────
(function() {{
  const sel = document.getElementById("regionalCategory");
  D.meta.categories.forEach(c => {{
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c.charAt(0).toUpperCase() + c.slice(1);
    sel.appendChild(opt);
  }});

  const ctx = document.getElementById("regionalChart").getContext("2d");
  let chart = null;

  function update() {{
    const cat = sel.value;
    const regions = Object.keys(D.regional).sort();
    const shortNames = regions.map(r => r.replace("region", "").replace("Autonomous region in Muslim Mindanao", "ARMM").replace("Cordillera Administrative", "CAR").replace("National Capital", "NCR").trim());

    const values = regions.map(r => {{
      if (cat === "_all") {{
        const cats = D.regional[r];
        const vals = Object.values(cats);
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      }}
      return D.regional[r][cat] || 0;
    }});

    if (chart) chart.destroy();
    chart = new Chart(ctx, {{
      type: "bar",
      data: {{
        labels: shortNames,
        datasets: [{{
          data: values,
          backgroundColor: values.map(v => {{
            const max = Math.max(...values);
            const ratio = v / max;
            return `rgba(99,102,241,${{0.3 + ratio * 0.7}})`;
          }}),
          borderColor: "#6366f1",
          borderWidth: 1,
          borderRadius: 4,
        }}],
      }},
      options: {{
        responsive: true,
        indexAxis: "y",
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => `PHP ${{ctx.parsed.x?.toFixed(2)}}/kg` }} }} }},
        scales: {{
          x: {{ title: {{ display: true, text: "Avg Price (PHP/kg)" }} }},
        }},
      }},
    }});
  }}

  sel.addEventListener("change", update);
  update();
}})();

// ─── Map ────────────────────────────────────────────────
(function() {{
  const map = L.map("map").setView([12.5, 122], 6);
  L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 18,
  }}).addTo(map);

  const prices = D.mapPoints.map(p => p.avg_price);
  const minP = Math.min(...prices), maxP = Math.max(...prices);

  D.mapPoints.forEach(p => {{
    const ratio = (p.avg_price - minP) / (maxP - minP || 1);
    const radius = 5 + ratio * 20;
    const color = ratio > 0.66 ? "#ef4444" : ratio > 0.33 ? "#f59e0b" : "#22c55e";

    L.circleMarker([p.lat, p.lng], {{
      radius: radius,
      color: color,
      fillColor: color,
      fillOpacity: 0.5,
      weight: 1,
    }}).addTo(map).bindPopup(`
      <strong style="font-size:14px">${{p.name}}</strong><br>
      <span style="color:#888">${{p.province}}, ${{p.region}}</span><br>
      <strong>Avg Price:</strong> PHP ${{p.avg_price}}/kg<br>
      <strong>Records:</strong> ${{p.records.toLocaleString()}}
    `);
  }});
}})();

// ─── Movers ─────────────────────────────────────────────
(function() {{
  const upEl = document.getElementById("moversUp");
  const downEl = document.getElementById("moversDown");
  const up = D.movers.filter(m => m.change_pct > 0).slice(0, 10);
  const down = D.movers.filter(m => m.change_pct < 0).reverse().slice(0, 10);

  function renderMovers(items, el, isUp) {{
    el.innerHTML = items.map(m => `
      <div class="mover-item">
        <div>
          <div class="mover-name">${{m.commodity}}</div>
          <div class="mover-prices">PHP ${{m.price_2020}} &rarr; PHP ${{m.price_2023}}</div>
        </div>
        <span class="badge ${{isUp ? 'badge-up' : 'badge-down'}}">
          ${{isUp ? '+' : ''}}${{m.change_pct}}%
        </span>
      </div>
    `).join("");
  }}

  renderMovers(up, upEl, true);
  renderMovers(down, downEl, false);
}})();

// ─── Commodity Table ────────────────────────────────────
(function() {{
  const sel = document.getElementById("tableCategory");
  const search = document.getElementById("tableSearch");
  const tbody = document.getElementById("commTableBody");
  let data = [...D.commodityTable];
  let sortCol = "avg", sortAsc = false;

  D.meta.categories.forEach(c => {{
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c.charAt(0).toUpperCase() + c.slice(1);
    sel.appendChild(opt);
  }});

  function render() {{
    let filtered = data;
    const cat = sel.value;
    const q = search.value.toLowerCase();
    if (cat !== "_all") filtered = filtered.filter(r => r.category === cat);
    if (q) filtered = filtered.filter(r => r.name.toLowerCase().includes(q));

    filtered.sort((a, b) => {{
      let va = a[sortCol], vb = b[sortCol];
      if (typeof va === "string") {{ va = va.toLowerCase(); vb = vb.toLowerCase(); }}
      return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    }});

    tbody.innerHTML = filtered.map(r => `
      <tr>
        <td><strong>${{r.name}}</strong></td>
        <td style="color:var(--text2)">${{r.category}}</td>
        <td>PHP ${{r.avg.toFixed(2)}}</td>
        <td style="color:var(--green)">PHP ${{r.min.toFixed(2)}}</td>
        <td style="color:var(--red)">PHP ${{r.max.toFixed(2)}}</td>
        <td>${{r.records.toLocaleString()}}</td>
      </tr>
    `).join("");
  }}

  document.querySelectorAll("#commTable th").forEach(th => {{
    th.addEventListener("click", () => {{
      const col = th.dataset.col;
      if (sortCol === col) sortAsc = !sortAsc;
      else {{ sortCol = col; sortAsc = true; }}
      render();
    }});
  }});

  sel.addEventListener("change", render);
  search.addEventListener("input", render);
  render();
}})();
</script>
</body>
</html>"""

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n{'=' * 60}")
print(f"  Dashboard saved to: {OUTPUT_PATH}")
print(f"  Open in browser to view!")
print(f"{'=' * 60}")
