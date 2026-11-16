#!/usr/bin/env python3
"""Run a local mock experiment to verify the full pipeline.

This uses simulated model responses — NO Sheshnag access required.
Mock results are clearly labelled and stored separately.

Usage:
    python scripts/run_mock.py [--questions 20] [--generations 10]
"""

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.batch import generate_batch_jsonl, validate_batch_file
from src.inference_budget.parsing import extract_model_answer, normalize_answer
from src.inference_budget.evaluation import (
    parse_sheshnag_results,
    check_result_integrity,
    run_evaluation,
    compute_accuracy,
)
from src.inference_budget.analysis import (
    bootstrap_accuracy_ci,
    compute_marginal_gains,
    create_accuracy_plot,
    create_marginal_gain_plot,
    create_budget_tradeoff_plot,
    generate_summary_json,
)


def generate_mock_response(gold_answer: str, rng: random.Random) -> str:
    """Generate a mock model response.
    
    Simulates a model that gets the answer right ~70% of the time,
    with some variation in response format.
    """
    # 70% chance of correct answer
    if rng.random() < 0.70:
        answer = gold_answer
    else:
        # Generate a wrong answer
        try:
            num = float(gold_answer.replace(",", ""))
            offset = rng.choice([-2, -1, 1, 2, 5, 10])
            answer = str(int(num + offset))
        except ValueError:
            answer = str(rng.randint(1, 100))
    
    # Vary response format
    fmt = rng.choice(["standard", "space", "hash", "boxed", "unparseable"])
    
    reasoning = (
        f"Let me work through this step by step.\n"
        f"First, I need to analyze the problem carefully.\n"
        f"After calculating, I find the answer is {answer}.\n"
    )
    
    if fmt == "standard":
        return reasoning + f"\nFINAL_ANSWER: {answer}"
    elif fmt == "space":
        return reasoning + f"\nFINAL ANSWER: {answer}"
    elif fmt == "hash":
        return reasoning + f"\n#### {answer}"
    elif fmt == "boxed":
        return reasoning + f"\n\\boxed{{{answer}}}"
    else:
        # ~5% unparseable
        if rng.random() < 0.3:
            return "I'm not sure about this problem."
        return reasoning + f"\nThe answer is {answer}"


def main():
    parser = argparse.ArgumentParser(
        description="Run a mock experiment (no Sheshnag required)."
    )
    parser.add_argument(
        "--questions", type=int, default=20,
        help="Number of mock questions (default: 20).",
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Generations per question (default: 50).",
    )
    args = parser.parse_args()
    
    base_config = load_config()
    mock_dir = resolve_path("results/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a config copy with sample_size=questions, generations_per_question=generations,
    # and budgets <= generations
    config = copy.deepcopy(base_config)
    config["dataset"]["sample_size"] = args.questions
    config["experiment"]["generations_per_question"] = args.generations
    filtered_budgets = [
        b for b in base_config["experiment"]["inference_budgets"]
        if b <= args.generations
    ]
    if not filtered_budgets:
        filtered_budgets = [args.generations]
    config["experiment"]["inference_budgets"] = filtered_budgets
    
    print(f"{'='*60}")
    print(f"MOCK EXPERIMENT")
    print(f"{'='*60}")
    print(f"Questions: {args.questions}")
    print(f"Generations/question: {args.generations}")
    print(f"Inference budgets: {config['experiment']['inference_budgets']}")
    print(f"Total requests: {args.questions * args.generations}")
    print(f"Output directory: {mock_dir}")
    print()
    
    # Step 1: Create mock dataset
    rng = random.Random(42)
    mock_questions = []
    for i in range(1, args.questions + 1):
        gold = str(rng.randint(1, 200))
        mock_questions.append({
            "question_id": i,
            "original_index": i - 1,
            "question": f"Mock math problem {i}: What is the answer?",
            "full_answer": f"The reasoning leads to #### {gold}",
            "gold_answer": gold,
        })
    
    dataset_df = pd.DataFrame(mock_questions)
    dataset_path = mock_dir / "mock_dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)
    print(f"1. Dataset generated: {len(dataset_df)} questions -> {dataset_path}")
    
    # Step 2: Generate batch JSONL
    mock_batch_path = mock_dir / "mock_batch.jsonl"
    mock_model_id = "mock-model-v1"
    generate_batch_jsonl(
        dataset=dataset_df,
        model_id=mock_model_id,
        config=config,
        output_path=mock_batch_path,
    )
    print(f"2. Batch JSONL generated -> {mock_batch_path}")
    
    # Step 3: Validate batch file
    validation_errors = validate_batch_file(mock_batch_path, config)
    if validation_errors:
        print(f"ERROR: Mock batch validation failed with {len(validation_errors)} error(s):")
        for err in validation_errors:
            print(f"  - {err}")
        sys.exit(1)
    print(f"3. Batch file validated successfully ({args.questions * args.generations} lines).")
    
    # Step 4: Fake model answering by custom_id with results written in SHUFFLED order
    print("4. Generating fake model answers by custom_id...")
    gold_by_qid = dict(zip(dataset_df["question_id"], dataset_df["gold_answer"]))
    
    mock_results = []
    gen_rng = random.Random(123)
    
    with open(mock_batch_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            req = json.loads(line)
            cid = req["custom_id"]
            
            m = re.match(r'^gsm8k_q(\d+)_g(\d+)$', cid)
            q_id = int(m.group(1)) if m else 1
            gold = str(gold_by_qid.get(q_id, "42"))
            
            response_text = generate_mock_response(gold, gen_rng)
            mock_results.append({
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {"content": response_text}
                        }],
                        "usage": {
                            "prompt_tokens": 150,
                            "completion_tokens": 100,
                            "total_tokens": 250,
                        },
                    }
                },
                "error": None,
            })
    
    # Write results in SHUFFLED order
    shuffle_rng = random.Random(999)
    shuffle_rng.shuffle(mock_results)
    
    mock_results_path = mock_dir / "mock_results.jsonl"
    with open(mock_results_path, "w", encoding="utf-8") as f:
        for r in mock_results:
            f.write(json.dumps(r) + "\n")
    print(f"   Saved {len(mock_results)} mock results in shuffled order -> {mock_results_path}")
    
    # Step 5: Parse results
    generations = parse_sheshnag_results(mock_results_path, config)
    mock_generations_path = mock_dir / "mock_generations.csv"
    generations.to_csv(mock_generations_path, index=False)
    
    total = len(generations)
    parsed = generations["parse_success"].sum()
    print(f"5. Parsed: {parsed}/{total} successfully parsed ({parsed/total*100:.1f}%) -> {mock_generations_path}")
    
    # Step 6: Integrity check
    check_result_integrity(generations, dataset_df, config)
    print("6. Result integrity check passed.")
    
    # Step 7: Run evaluation
    wide_df, long_df = run_evaluation(generations, dataset_df, config)
    wide_df.to_csv(mock_dir / "mock_evaluation_wide.csv", index=False)
    long_df.to_csv(mock_dir / "mock_evaluation_long.csv", index=False)
    print("7. Evaluation completed.")
    
    # Compute accuracy
    summary = compute_accuracy(long_df)
    
    print(f"\n{'='*50}")
    print(f"MOCK ACCURACY RESULTS")
    print(f"{'='*50}")
    print(f"{'N':>5} | {'Correct':>8} | {'Total':>6} | {'Accuracy':>10}")
    print(f"{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")
    for _, row in summary.iterrows():
        print(
            f"{int(row['N']):>5} | {int(row['correct']):>8} | "
            f"{int(row['total']):>6} | {row['accuracy']:>9.2%}"
        )
    print(f"{'='*50}")
    
    # Step 8: Analysis and plots
    print("\n8. Running analysis and plotting...")
    ci_data = {}
    for n in config["experiment"]["inference_budgets"]:
        acc, lower, upper = bootstrap_accuracy_ci(
            long_df, n,
            n_bootstrap=1000,
            seed=42,
        )
        ci_data[n] = (acc, lower, upper)
        print(f"  N={n:>2}: {acc:.4f} [{lower:.4f}, {upper:.4f}]")
    
    gains = compute_marginal_gains(summary)
    print("\nMarginal gains:")
    for _, row in gains.iterrows():
        print(
            f"  N={int(row['from_N'])}->{int(row['to_N'])}: "
            f"{row['delta_accuracy']:+.4f}"
        )
    
    figures_dir = mock_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    generate_summary_json(
        summary, gains, ci_data,
        output_path=mock_dir / "mock_summary.json",
    )
    summary.to_csv(mock_dir / "mock_summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"MOCK EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"All mock outputs saved to: {mock_dir}")
    print(f"Pipeline verified successfully end-to-end.\n")


if __name__ == "__main__":
    main()
