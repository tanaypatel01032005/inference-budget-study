#!/usr/bin/env python3
"""Run full analysis: confidence intervals, marginal gains, and plots.

Usage:
    python scripts/analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.evaluation import compute_accuracy
from src.inference_budget.analysis import (
    bootstrap_accuracy_ci,
    compute_marginal_gains,
    create_accuracy_plot,
    create_marginal_gain_plot,
    create_budget_tradeoff_plot,
    generate_summary_json,
)


def main():
    config = load_config()
    analysis_config = config.get("analysis", {})
    
    # Load evaluation results
    long_path = resolve_path(config["paths"]["results_final"]).parent / "evaluation_long.csv"
    if not long_path.exists():
        print(f"ERROR: Evaluation results not found: {long_path}")
        print("Run: python scripts/evaluate.py")
        sys.exit(1)
    
    long_df = pd.read_csv(long_path)
    print(f"Loaded {len(long_df)} evaluation records")
    
    # Compute accuracy summary
    summary = compute_accuracy(long_df)
    print("\nAccuracy Summary:")
    print(summary.to_string(index=False))
    
    # Bootstrap confidence intervals
    budgets = config["experiment"]["inference_budgets"]
    n_bootstrap = analysis_config.get("bootstrap_samples", 10000)
    confidence = analysis_config.get("confidence_level", 0.95)
    bootstrap_seed = analysis_config.get("bootstrap_seed", 42)
    
    print(f"\nComputing {confidence*100:.0f}% confidence intervals ({n_bootstrap} bootstrap samples)...")
    ci_data = {}
    for n in budgets:
        acc, lower, upper = bootstrap_accuracy_ci(
            long_df, n,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        ci_data[n] = (acc, lower, upper)
        print(f"  N={n:>2}: {acc:.4f} [{lower:.4f}, {upper:.4f}]")
    
    # Marginal gains
    gains = compute_marginal_gains(summary)
    print("\nMarginal Gains:")
    print(gains.to_string(index=False))
    
    # Generate plots
    figures_dir = resolve_path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    create_accuracy_plot(
        summary, ci_data,
        output_path=figures_dir / "accuracy_vs_generations.png",
    )
    create_marginal_gain_plot(
        gains,
        output_path=figures_dir / "marginal_gain.png",
    )
    create_budget_tradeoff_plot(
        summary, ci_data,
        output_path=figures_dir / "accuracy_vs_budget.png",
    )
    
    # Save summary CSV
    analysis_dir = resolve_path(config["paths"]["results_analysis"])
    summary_with_ci = summary.copy()
    summary_with_ci["ci_lower"] = [ci_data[n][1] for n in summary_with_ci["N"]]
    summary_with_ci["ci_upper"] = [ci_data[n][2] for n in summary_with_ci["N"]]
    summary_with_ci.to_csv(analysis_dir / "summary.csv", index=False)
    
    # Save gains
    gains.to_csv(analysis_dir / "marginal_gains.csv", index=False)
    
    # Save JSON summary
    generate_summary_json(
        summary, gains, ci_data,
        output_path=analysis_dir / "summary.json",
    )
    
    # Update results summary markdown
    update_results_summary(summary_with_ci, gains, ci_data, config)
    
    print(f"\nAnalysis complete!")
    print(f"  Figures: {figures_dir}")
    print(f"  Summary: {analysis_dir}")


def update_results_summary(summary_df, gains_df, ci_data, config):
    """Update the results summary markdown file."""
    report_path = resolve_path("reports/results_summary.md")
    
    lines = ["# Results Summary\n"]
    lines.append(f"*Generated automatically by analyze.py*\n")
    
    lines.append("\n## Accuracy by Inference Budget\n")
    lines.append("| N | Correct | Total | Accuracy | 95% CI Lower | 95% CI Upper |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        n = int(row["N"])
        lines.append(
            f"| {n} | {int(row['correct'])} | {int(row['total'])} | "
            f"{row['accuracy']:.4f} | {row['ci_lower']:.4f} | {row['ci_upper']:.4f} |"
        )
    
    lines.append("\n## Marginal Gains\n")
    lines.append("| From N | To N | Δ Accuracy | Gain per Generation |")
    lines.append("|---:|---:|---:|---:|")
    for _, row in gains_df.iterrows():
        lines.append(
            f"| {int(row['from_N'])} | {int(row['to_N'])} | "
            f"{row['delta_accuracy']:+.4f} | {row['accuracy_gain_per_generation']:.6f} |"
        )
    
    lines.append("")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Updated results summary: {report_path}")


if __name__ == "__main__":
    main()
