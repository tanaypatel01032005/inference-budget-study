#!/usr/bin/env python3
"""Run evaluation on Sheshnag results.

Usage:
    python scripts/evaluate.py [--results-file PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.dataset import load_processed_dataset
from src.inference_budget.evaluation import (
    parse_sheshnag_results,
    run_evaluation,
    compute_accuracy,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on Sheshnag results."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to raw results JSONL file.",
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = resolve_path(config["paths"]["results_raw"])
    
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run the Sheshnag submission first, then download results.")
        sys.exit(1)
    
    # Load dataset
    dataset = load_processed_dataset(config)
    print(f"Loaded {len(dataset)} questions")
    
    # Parse results
    print(f"\nParsing results from {results_path}...")
    generations = parse_sheshnag_results(results_path, config)
    
    # Save parsed generations
    parsed_path = resolve_path(config["paths"]["results_parsed"])
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_csv(parsed_path, index=False)
    print(f"Saved parsed generations to {parsed_path}")
    
    # Parse success stats
    total_gens = len(generations)
    successful = generations["parse_success"].sum()
    print(f"\nParsing stats:")
    print(f"  Total generations: {total_gens}")
    print(f"  Successfully parsed: {successful} ({successful/total_gens*100:.1f}%)")
    print(f"  Unparseable: {total_gens - successful} ({(total_gens-successful)/total_gens*100:.1f}%)")
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    wide_df, long_df = run_evaluation(generations, dataset, config)
    
    # Save results
    final_path = resolve_path(config["paths"]["results_final"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(final_path, index=False)
    
    long_path = final_path.parent / "evaluation_long.csv"
    long_df.to_csv(long_path, index=False)
    
    print(f"Saved evaluation results:")
    print(f"  Wide format: {final_path}")
    print(f"  Long format: {long_path}")
    
    # Compute and display accuracy
    summary = compute_accuracy(long_df)
    
    print(f"\n{'='*50}")
    print(f"ACCURACY RESULTS")
    print(f"{'='*50}")
    print(f"{'N':>5} | {'Correct':>8} | {'Total':>6} | {'Accuracy':>10}")
    print(f"{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")
    for _, row in summary.iterrows():
        print(
            f"{int(row['N']):>5} | {int(row['correct']):>8} | "
            f"{int(row['total']):>6} | {row['accuracy']:>9.2%}"
        )
    print(f"{'='*50}")
    
    # Save summary
    summary_path = resolve_path(config["paths"]["results_analysis"]) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(f"\nNext step: python scripts/analyze.py")


if __name__ == "__main__":
    main()
