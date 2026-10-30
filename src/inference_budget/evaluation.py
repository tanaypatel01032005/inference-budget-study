"""Evaluation logic: parsing results, voting, and accuracy calculation."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import load_config, resolve_path
from .parsing import extract_model_answer, normalize_answer, answers_match
from .voting import evaluate_at_n


def parse_sheshnag_results(
    results_path: Path,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Parse raw Sheshnag JSONL results into a structured DataFrame.
    
    Args:
        results_path: Path to the raw results JSONL file.
        config: Configuration dictionary.
    
    Returns:
        DataFrame with parsed generations.
    """
    import re
    
    records = []
    
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            
            # Parse question_id and generation_id from custom_id
            match = re.match(r'gsm8k_q(\d+)_g(\d+)', custom_id)
            if not match:
                print(f"Warning: Unrecognized custom_id format: {custom_id}")
                continue
            
            question_id = int(match.group(1))
            generation_id = int(match.group(2))
            
            # Extract response
            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            
            raw_response = ""
            if choices:
                raw_response = choices[0].get("message", {}).get("content", "")
            
            # Extract answer
            parsed_answer, extraction_method = extract_model_answer(raw_response)
            parse_success = parsed_answer is not None
            
            # Extract usage info if available
            usage = response_body.get("usage", {})
            
            record = {
                "question_id": question_id,
                "generation_id": generation_id,
                "custom_id": custom_id,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "answer_extraction_method": extraction_method,
                "parse_success": parse_success,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} generation results")
    return df


def run_evaluation(
    generations_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full evaluation across all inference budgets.
    
    For each question:
    1. Retrieve its 50 generations
    2. Deterministically shuffle them
    3. For each N in inference_budgets:
        - Select first N answers
        - Perform majority vote
        - Compare to gold answer
    
    Args:
        generations_df: DataFrame with parsed generations.
        dataset_df: DataFrame with questions and gold answers.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (wide_results_df, long_results_df).
    """
    if config is None:
        config = load_config()
    
    budgets = config["experiment"]["inference_budgets"]
    shuffle_seed = config["experiment"]["shuffle_seed"]
    
    wide_records = []
    long_records = []
    
    for _, q_row in dataset_df.iterrows():
        q_id = int(q_row["question_id"])
        gold = normalize_answer(str(q_row["gold_answer"]))
        
        # Get this question's generations
        q_gens = generations_df[
            generations_df["question_id"] == q_id
        ].copy()
        
        if len(q_gens) == 0:
            print(f"Warning: No generations found for question {q_id}")
            continue
        
        # Deterministically shuffle generations
        rng = random.Random(shuffle_seed + q_id)
        gen_indices = list(q_gens.index)
        rng.shuffle(gen_indices)
        q_gens = q_gens.loc[gen_indices].reset_index(drop=True)
        
        # Get all answers in shuffled order
        all_answers = q_gens["parsed_answer"].tolist()
        
        wide_record = {
            "question_id": q_id,
            "gold_answer": gold,
        }
        
        for n in budgets:
            predicted, vote_count, valid_count, vote_dist = evaluate_at_n(
                all_answers, n
            )
            correct = answers_match(predicted, gold)
            
            wide_record[f"answer_N{n}"] = predicted
            wide_record[f"correct_N{n}"] = correct
            wide_record[f"valid_gens_N{n}"] = valid_count
            wide_record[f"votes_N{n}"] = vote_count
            
            long_records.append({
                "question_id": q_id,
                "N": n,
                "predicted_answer": predicted,
                "gold_answer": gold,
                "correct": correct,
                "valid_generations": valid_count,
                "total_votes": vote_count,
                "unparseable": n - valid_count,
            })
        
        wide_records.append(wide_record)
    
    wide_df = pd.DataFrame(wide_records)
    long_df = pd.DataFrame(long_records)
    
    return wide_df, long_df


def compute_accuracy(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy for each inference budget N.
    
    Args:
        long_df: Long-form evaluation results.
    
    Returns:
        DataFrame with accuracy per N.
    """
    summary = long_df.groupby("N").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
    ).reset_index()
    
    summary["accuracy"] = summary["correct"] / summary["total"]
    summary["incorrect"] = summary["total"] - summary["correct"]
    
    return summary[["N", "total", "correct", "incorrect", "accuracy"]]
