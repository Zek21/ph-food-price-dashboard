"""
Comprehensive model comparison report generator.
# signed: delta

Reads model_comparison.json (individual models) and ensemble_predictions.json
(ensemble stacking), then generates a unified model_report.json with:
  - Per-model metrics (MAPE, MAE, RMSE, R2, bias)
  - Best model per commodity
  - Regional accuracy heatmap data
  - Feature importance rankings (for tree-based models)
  - Ensemble vs individual comparison
  - Recommendations and summary statistics

Usage:
    python model_report.py                    # Generate full report
    python model_report.py --output report.json  # Custom output path
    python model_report.py --format markdown  # Also generate markdown summary
"""
# signed: delta

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    _SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

MODEL_COMPARISON_PATH = _SCRIPT_DIR / "model_comparison.json"
ENSEMBLE_PATH = _SCRIPT_DIR / "ensemble_predictions.json"
REPORT_OUTPUT_PATH = _SCRIPT_DIR / "model_report.json"


def load_model_comparison() -> dict:
    """Load individual model comparison results."""
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"model_comparison.json not found at {MODEL_COMPARISON_PATH}. "
            "Run retrain_model.py first."
        )
    with open(MODEL_COMPARISON_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_ensemble_results() -> dict | None:
    """Load ensemble predictions if available."""
    if not ENSEMBLE_PATH.exists():
        return None
    with open(ENSEMBLE_PATH, encoding="utf-8") as f:
        return json.load(f)
    # signed: delta


def compute_per_model_metrics(mc: dict) -> dict:
    """Extract and enrich per-model metrics from model_comparison.json.

    Returns: {model_name: {mape, mae, rmse, r2, bias, n_val, rank}}
    """
    # signed: delta
    models = mc.get("models", [])
    overall = mc.get("overall", {})

    metrics = {}
    for model_name in models:
        m = overall.get(model_name, {})
        metrics[model_name] = {
            "mape": m.get("mape", None),
            "mae": m.get("mae", None),
            "r2": m.get("r2", None),
            "bias": m.get("bias", None),
            "n_val": m.get("n_val", 0),
        }

    # Rank by MAPE (lower is better)
    sorted_models = sorted(
        metrics.items(),
        key=lambda x: x[1].get("mape", float("inf")),
    )
    for rank, (name, _) in enumerate(sorted_models, 1):
        metrics[name]["mape_rank"] = rank

    return metrics


def compute_best_model_per_commodity(mc: dict) -> dict:
    """Determine which model performs best for each commodity.

    Returns: {commodity: {best_model, mape, mae, all_models: {model: metrics}}}
    """
    # signed: delta
    comm_comparison = mc.get("commComparison", {})
    models = mc.get("models", [])

    best_per_comm = {}
    for comm, model_metrics in comm_comparison.items():
        best_model = None
        best_mape = float("inf")
        all_models = {}

        for model_name in models:
            m = model_metrics.get(model_name, {})
            mape = m.get("mape")
            if mape is not None:
                all_models[model_name] = {
                    "mape": mape,
                    "mae": m.get("mae"),
                    "bias": m.get("bias"),
                }
                if mape < best_mape:
                    best_mape = mape
                    best_model = model_name

        best_per_comm[comm] = {
            "best_model": best_model,
            "best_mape": round(best_mape, 2) if best_mape < float("inf") else None,
            "all_models": all_models,
        }

    return best_per_comm


def compute_regional_accuracy(mc: dict) -> dict:
    """Build regional accuracy heatmap data.

    Aggregates validation predictions by region for each model.
    Returns: {model: {region: {mape, mae, n_samples}}}
    """
    # signed: delta
    comm_comparison = mc.get("commComparison", {})
    models = mc.get("models", [])

    # Collect per-region data from commComparison validation details
    regional_data = {m: defaultdict(lambda: {"actual": [], "pred": []}) for m in models}

    for comm, model_metrics in comm_comparison.items():
        for model_name in models:
            m = model_metrics.get(model_name, {})
            val_regions = m.get("val_regions", [])
            val_actual = m.get("val_actual", [])
            val_pred = m.get("val_pred", [])

            for i in range(min(len(val_regions), len(val_actual), len(val_pred))):
                region = val_regions[i]
                regional_data[model_name][region]["actual"].append(val_actual[i])
                regional_data[model_name][region]["pred"].append(val_pred[i])

    # Compute per-region metrics
    regional_metrics = {}
    for model_name in models:
        model_regions = {}
        for region, data in regional_data[model_name].items():
            actual = np.array(data["actual"])
            pred = np.array(data["pred"])
            if len(actual) > 0 and np.all(actual > 0):
                mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
                mae = float(np.mean(np.abs(actual - pred)))
                model_regions[region] = {
                    "mape": round(mape, 2),
                    "mae": round(mae, 2),
                    "n_samples": len(actual),
                }
        regional_metrics[model_name] = model_regions

    return regional_metrics


def compute_feature_importance(mc: dict) -> dict:
    """Extract feature importance rankings from variant search data.

    Since we don't have fitted models here, we derive importance from
    parameter sensitivity: how much does MAPE change across variants.
    Returns: {model: {parameter: importance_score}}
    """
    # signed: delta
    variant_search = mc.get("variantSearch", {})
    feature_importance = {}

    for model_name, vs in variant_search.items():
        param_grid = vs.get("parameter_grid", [])
        if not param_grid:
            continue

        # Compute parameter variation as a proxy for importance
        all_params = set()
        for pg in param_grid:
            all_params.update(pg.keys())

        param_variation = {}
        for param in all_params:
            values = [pg.get(param) for pg in param_grid if pg.get(param) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                # Coefficient of variation as importance proxy
                arr = np.array(values, dtype=float)
                if arr.mean() != 0:
                    cv = float(arr.std() / abs(arr.mean()))
                    param_variation[param] = round(cv, 3)

        # Rank by variation (more variation = more tuned = more important)
        sorted_params = sorted(
            param_variation.items(), key=lambda x: x[1], reverse=True
        )
        feature_importance[model_name] = {
            p: {"variation": v, "rank": i + 1}
            for i, (p, v) in enumerate(sorted_params)
        }

    return feature_importance


def build_summary_statistics(
    per_model: dict,
    best_per_comm: dict,
    ensemble: dict | None,
) -> dict:
    """Build high-level summary statistics."""
    # signed: delta
    model_counts = defaultdict(int)
    for comm_data in best_per_comm.values():
        best = comm_data.get("best_model")
        if best:
            model_counts[best] += 1

    # Average MAPE across models
    mapes = [m["mape"] for m in per_model.values() if m.get("mape") is not None]
    avg_mape = round(np.mean(mapes), 2) if mapes else None

    best_overall_model = min(
        per_model.items(),
        key=lambda x: x[1].get("mape", float("inf")),
    )

    summary = {
        "total_models": len(per_model),
        "total_commodities": len(best_per_comm),
        "average_mape_across_models": avg_mape,
        "best_overall_model": best_overall_model[0],
        "best_overall_mape": best_overall_model[1].get("mape"),
        "model_win_counts": dict(model_counts),
    }

    if ensemble:
        ens_eval = ensemble.get("evaluation", {})
        ens_overall = ens_eval.get("ensemble_overall", {})
        if ens_overall:
            summary["ensemble_mape"] = ens_overall.get("mape")
            summary["ensemble_r2"] = ens_overall.get("r2")
            summary["ensemble_beats_average"] = (
                ens_overall.get("mape", float("inf")) < avg_mape
                if avg_mape
                else None
            )

    return summary


def build_recommendations(
    per_model: dict,
    best_per_comm: dict,
    ensemble: dict | None,
) -> list:
    """Generate actionable recommendations based on analysis."""
    # signed: delta
    recs = []

    # Best individual model
    best_model = min(
        per_model.items(),
        key=lambda x: x[1].get("mape", float("inf")),
    )
    recs.append({
        "priority": 1,
        "recommendation": f"Use {best_model[0]} as primary model (MAPE: {best_model[1].get('mape')}%)",
        "rationale": "Lowest MAPE across all commodities on validation set.",
    })

    # Ensemble recommendation
    if ensemble:
        ens_mape = (
            ensemble.get("evaluation", {})
            .get("ensemble_overall", {})
            .get("mape")
        )
        if ens_mape is not None:
            best_ind_mape = best_model[1].get("mape", float("inf"))
            if ens_mape < best_ind_mape:
                recs.append({
                    "priority": 2,
                    "recommendation": (
                        f"Adopt ensemble stacking (MAPE: {ens_mape}%) -- "
                        f"outperforms best individual by {round(best_ind_mape - ens_mape, 2)}%"
                    ),
                    "rationale": "Stacking combines base model strengths, reducing prediction variance.",
                })
            else:
                recs.append({
                    "priority": 3,
                    "recommendation": (
                        f"Ensemble stacking (MAPE: {ens_mape}%) does not beat "
                        f"best individual ({best_model[0]}: {best_ind_mape}%). "
                        "Consider tuning meta-learner or adding base models."
                    ),
                    "rationale": "Ensemble overhead not justified without accuracy improvement.",
                })

    # Per-commodity best model variation
    model_wins = defaultdict(int)
    for comm_data in best_per_comm.values():
        m = comm_data.get("best_model")
        if m:
            model_wins[m] += 1

    if len(model_wins) > 1:
        recs.append({
            "priority": 4,
            "recommendation": (
                "Consider commodity-specific model routing -- different models "
                f"win for different commodities: {dict(model_wins)}"
            ),
            "rationale": "No single model dominates all commodities; adaptive routing improves accuracy.",
        })

    return recs


def generate_report(output_path: Path | None = None) -> dict:
    """Generate the full model comparison report."""
    # signed: delta
    print("=" * 65)
    print("  Model Comparison Report Generator")
    print("=" * 65)

    mc = load_model_comparison()
    ensemble = load_ensemble_results()

    print(f"\n   Models: {len(mc.get('models', []))}")
    print(f"   Ensemble available: {'Yes' if ensemble else 'No'}")

    # Compute all report sections
    per_model = compute_per_model_metrics(mc)
    best_per_comm = compute_best_model_per_commodity(mc)
    regional_accuracy = compute_regional_accuracy(mc)
    feature_importance = compute_feature_importance(mc)
    summary = build_summary_statistics(per_model, best_per_comm, ensemble)
    recommendations = build_recommendations(per_model, best_per_comm, ensemble)

    report = {
        "report_version": "1.0",
        "summary": summary,
        "per_model_metrics": per_model,
        "best_model_per_commodity": best_per_comm,
        "regional_accuracy_heatmap": regional_accuracy,
        "feature_importance_rankings": feature_importance,
        "recommendations": recommendations,
        "data_sources": {
            "model_comparison": str(MODEL_COMPARISON_PATH),
            "ensemble_predictions": str(ENSEMBLE_PATH) if ensemble else None,
        },
    }

    # Add ensemble section if available
    if ensemble:
        report["ensemble"] = {
            "model_type": ensemble.get("model"),
            "base_models": ensemble.get("base_models"),
            "meta_learner": ensemble.get("meta_learner"),
            "evaluation": ensemble.get("evaluation"),
        }

    out = output_path or REPORT_OUTPUT_PATH
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n   Report saved to {out}")
    print(f"   File size: {out.stat().st_size / 1024:.1f} KB")

    # Print summary
    print(f"\n   --- Summary ---")
    print(f"   Best model: {summary['best_overall_model']} (MAPE: {summary['best_overall_mape']}%)")
    print(f"   Avg MAPE across models: {summary['average_mape_across_models']}%")
    if summary.get("ensemble_mape") is not None:
        print(f"   Ensemble MAPE: {summary['ensemble_mape']}%")
    print(f"   Commodities analyzed: {summary['total_commodities']}")
    print(f"   Recommendations: {len(recommendations)}")

    return report


def generate_markdown(report: dict, output_path: Path | None = None) -> str:
    """Generate a markdown summary of the report."""
    # signed: delta
    md = ["# Philippine Food Price -- Model Comparison Report\n"]

    summary = report.get("summary", {})
    md.append("## Summary\n")
    md.append(f"- **Best Model:** {summary.get('best_overall_model', 'N/A')}")
    md.append(f"- **Best MAPE:** {summary.get('best_overall_mape', 'N/A')}%")
    md.append(f"- **Total Models:** {summary.get('total_models', 0)}")
    md.append(f"- **Commodities:** {summary.get('total_commodities', 0)}")
    if summary.get("ensemble_mape") is not None:
        md.append(f"- **Ensemble MAPE:** {summary['ensemble_mape']}%")
    md.append("")

    # Per-model table
    md.append("## Per-Model Metrics\n")
    md.append("| Model | MAPE | MAE | R2 | Bias | Rank |")
    md.append("|-------|------|-----|------|------|------|")
    per_model = report.get("per_model_metrics", {})
    for name, m in sorted(per_model.items(), key=lambda x: x[1].get("mape_rank", 99)):
        md.append(
            f"| {name} | {m.get('mape', 'N/A')}% | "
            f"{m.get('mae', 'N/A')} | {m.get('r2', 'N/A')} | "
            f"{m.get('bias', 'N/A')}% | {m.get('mape_rank', '-')} |"
        )
    md.append("")

    # Recommendations
    md.append("## Recommendations\n")
    for rec in report.get("recommendations", []):
        md.append(f"**{rec['priority']}.** {rec['recommendation']}")
        md.append(f"   *{rec['rationale']}*\n")

    text = "\n".join(md)

    if output_path:
        output_path.write_text(text, encoding="utf-8")
        print(f"   Markdown saved to {output_path}")

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive model comparison report"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORT_OUTPUT_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    report = generate_report(Path(args.output))

    if args.format in ("markdown", "both"):
        md_path = Path(args.output).with_suffix(".md")
        generate_markdown(report, md_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
