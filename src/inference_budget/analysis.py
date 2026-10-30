"""Analysis: confidence intervals, marginal gains, and visualization."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_path


def bootstrap_accuracy_ci(
    long_df: pd.DataFrame,
    n_value: int,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval for accuracy at a given N.
    
    The experimental unit is the question (not individual generations).
    
    Args:
        long_df: Long-form evaluation results.
        n_value: The inference budget N to analyze.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95).
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (accuracy, ci_lower, ci_upper).
    """
    subset = long_df[long_df["N"] == n_value]
    correctness = subset["correct"].values.astype(float)
    
    rng = np.random.RandomState(seed)
    n_questions = len(correctness)
    
    boot_accuracies = []
    for _ in range(n_bootstrap):
        sample = rng.choice(correctness, size=n_questions, replace=True)
        boot_accuracies.append(sample.mean())
    
    boot_accuracies = np.array(boot_accuracies)
    accuracy = correctness.mean()
    
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(boot_accuracies, 100 * (1 - alpha / 2))
    
    return float(accuracy), float(ci_lower), float(ci_upper)


def compute_marginal_gains(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute marginal accuracy improvement between consecutive N values.
    
    Args:
        summary_df: DataFrame with N and accuracy columns.
    
    Returns:
        DataFrame with marginal gains.
    """
    sorted_df = summary_df.sort_values("N").reset_index(drop=True)
    
    gains = []
    for i in range(1, len(sorted_df)):
        current = sorted_df.iloc[i]
        previous = sorted_df.iloc[i - 1]
        delta = current["accuracy"] - previous["accuracy"]
        extra_gens = current["N"] - previous["N"]
        gain_per_gen = delta / extra_gens if extra_gens > 0 else 0
        
        gains.append({
            "from_N": int(previous["N"]),
            "to_N": int(current["N"]),
            "delta_accuracy": delta,
            "additional_generations": extra_gens,
            "accuracy_gain_per_generation": gain_per_gen,
        })
    
    return pd.DataFrame(gains)


def create_accuracy_plot(
    summary_df: pd.DataFrame,
    ci_data: Optional[Dict[int, Tuple[float, float, float]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Create the main accuracy vs. number of generations plot.
    
    Args:
        summary_df: DataFrame with N and accuracy columns.
        ci_data: Optional dict mapping N to (accuracy, ci_lower, ci_upper).
        output_path: Where to save the plot.
    
    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_df = summary_df.sort_values("N")
    n_values = sorted_df["N"].values
    accuracies = sorted_df["accuracy"].values * 100  # Convert to percentage
    
    if ci_data:
        ci_lower = np.array([ci_data[n][1] * 100 for n in n_values])
        ci_upper = np.array([ci_data[n][2] * 100 for n in n_values])
        yerr = np.array([accuracies - ci_lower, ci_upper - accuracies])
        ax.errorbar(
            n_values, accuracies, yerr=yerr,
            fmt='o-', capsize=5, capthick=2, linewidth=2,
            markersize=8, color='#2196F3', ecolor='#90CAF9',
            label='Accuracy (95% CI)',
        )
    else:
        ax.plot(
            n_values, accuracies, 'o-',
            linewidth=2, markersize=8, color='#2196F3',
            label='Accuracy',
        )
    
    ax.set_xlabel("Number of Generations (N)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Effect of Inference-Time Sampling on\nMathematical Reasoning Accuracy",
        fontsize=15, fontweight='bold',
    )
    ax.set_xticks(n_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(n_values) + 2)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path("results/analysis/figures/accuracy_vs_generations.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved accuracy plot to {output_path}")
    return output_path


def create_marginal_gain_plot(
    gains_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Create the marginal improvement plot.
    
    Args:
        gains_df: DataFrame with marginal gains.
        output_path: Where to save the plot.
    
    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [f"{int(r['from_N'])}->{int(r['to_N'])}" for _, r in gains_df.iterrows()]
    deltas = gains_df["delta_accuracy"].values * 100  # percentage points
    
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]
    
    bars = ax.bar(labels, deltas, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Generation Budget Transition", fontsize=13)
    ax.set_ylabel("Accuracy Improvement (pp)", fontsize=13)
    ax.set_title("Marginal Accuracy Gain per Budget Increase", fontsize=15, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., y_pos,
            f'{val:+.2f}',
            ha='center', va='bottom' if y_pos >= 0 else 'top',
            fontsize=10, fontweight='bold',
        )
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path("results/analysis/figures/marginal_gain.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved marginal gain plot to {output_path}")
    return output_path


def create_budget_tradeoff_plot(
    summary_df: pd.DataFrame,
    ci_data: Optional[Dict[int, Tuple[float, float, float]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Create accuracy vs. generation budget (cost proxy) plot.
    
    Args:
        summary_df: DataFrame with N and accuracy columns.
        ci_data: Optional confidence interval data.
        output_path: Where to save the plot.
    
    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_df = summary_df.sort_values("N")
    n_values = sorted_df["N"].values
    accuracies = sorted_df["accuracy"].values * 100
    
    # Generation budget = N * 1000 questions
    budgets = n_values * 1000
    
    if ci_data:
        ci_lower = np.array([ci_data[n][1] * 100 for n in n_values])
        ci_upper = np.array([ci_data[n][2] * 100 for n in n_values])
        yerr = np.array([accuracies - ci_lower, ci_upper - accuracies])
        ax.errorbar(
            budgets, accuracies, yerr=yerr,
            fmt='s-', capsize=5, capthick=2, linewidth=2,
            markersize=8, color='#FF9800', ecolor='#FFE0B2',
            label='Accuracy (95% CI)',
        )
    else:
        ax.plot(
            budgets, accuracies, 's-',
            linewidth=2, markersize=8, color='#FF9800',
            label='Accuracy',
        )
    
    ax.set_xlabel("Total Generation Budget (across 1,000 questions)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Accuracy vs. Computational Budget\n(Quality-Cost Trade-off)",
        fontsize=15, fontweight='bold',
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add N labels at each point
    for n, budget, acc in zip(n_values, budgets, accuracies):
        ax.annotate(
            f'N={n}', (budget, acc),
            textcoords="offset points", xytext=(10, 10),
            fontsize=9, fontweight='bold',
        )
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path("results/analysis/figures/accuracy_vs_budget.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved budget trade-off plot to {output_path}")
    return output_path


def generate_summary_json(
    summary_df: pd.DataFrame,
    gains_df: pd.DataFrame,
    ci_data: Dict[int, Tuple[float, float, float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a machine-readable JSON summary.
    
    Args:
        summary_df: Accuracy summary.
        gains_df: Marginal gains.
        ci_data: Confidence intervals.
        output_path: Where to save.
    
    Returns:
        Path to the saved JSON.
    """
    summary = {
        "accuracy_by_n": {},
        "marginal_gains": [],
        "confidence_intervals": {},
    }
    
    for _, row in summary_df.iterrows():
        n = int(row["N"])
        summary["accuracy_by_n"][str(n)] = {
            "N": n,
            "total": int(row["total"]),
            "correct": int(row["correct"]),
            "incorrect": int(row["incorrect"]),
            "accuracy": float(row["accuracy"]),
        }
    
    for _, row in gains_df.iterrows():
        summary["marginal_gains"].append({
            "from_N": int(row["from_N"]),
            "to_N": int(row["to_N"]),
            "delta_accuracy": float(row["delta_accuracy"]),
            "additional_generations": int(row["additional_generations"]),
            "accuracy_gain_per_generation": float(row["accuracy_gain_per_generation"]),
        })
    
    for n, (acc, lower, upper) in ci_data.items():
        summary["confidence_intervals"][str(n)] = {
            "accuracy": acc,
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": 0.95,
        }
    
    if output_path is None:
        output_path = Path("results/analysis/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary JSON to {output_path}")
    return output_path
