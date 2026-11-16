"""Evaluation logic: parsing results, voting, and accuracy calculation."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import load_config, resolve_path
from .parsing import extract_model_answer, normalize_answer, answers_match
from .voting import evaluate_at_n


class IncompleteResultsError(ValueError):
    """Raised when evaluation results fail integrity or completeness checks."""
    pass


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
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                records.append({
                    "question_id": -1,
                    "generation_id": -1,
                    "custom_id": "",
                    "request_ok": False,
                    "error": f"JSON decode error: {e}",
                    "raw_response": "",
                    "parsed_answer": None,
                    "answer_extraction_method": "unparseable",
                    "parse_success": False,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                })
                continue
            
            custom_id = result.get("custom_id", "")
            
            # Map only by custom_id (bad IDs -> question_id=-1)
            match = re.match(r'^gsm8k_q(\d+)_g(\d+)$', custom_id) if custom_id else None
            if match:
                question_id = int(match.group(1))
                generation_id = int(match.group(2))
            else:
                question_id = -1
                generation_id = -1
            
            # Check for failure conditions (response null / error set / status != 200)
            error_val = result.get("error")
            response = result.get("response")
            
            request_ok = True
            error_text = None
            
            if error_val is not None:
                request_ok = False
                error_text = json.dumps(error_val) if isinstance(error_val, (dict, list)) else str(error_val)
            elif response is None:
                request_ok = False
                error_text = "Response is null"
            elif isinstance(response, dict):
                status_code = response.get("status_code", 200)
                if status_code != 200:
                    request_ok = False
                    error_text = f"HTTP status {status_code}"
                else:
                    body = response.get("body")
                    if isinstance(body, dict) and body.get("error"):
                        request_ok = False
                        error_text = str(body.get("error"))
            
            raw_response = ""
            parsed_answer = None
            extraction_method = "unparseable"
            parse_success = False
            usage = {}
            
            if request_ok and isinstance(response, dict):
                response_body = response.get("body", {})
                if isinstance(response_body, dict):
                    choices = response_body.get("choices", [])
                    if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                        raw_response = choices[0].get("message", {}).get("content", "") or ""
                    parsed_answer, extraction_method = extract_model_answer(raw_response)
                    parse_success = parsed_answer is not None
                    usage = response_body.get("usage", {}) or {}
            
            record = {
                "question_id": question_id,
                "generation_id": generation_id,
                "custom_id": custom_id,
                "request_ok": request_ok,
                "error": error_text,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "answer_extraction_method": extraction_method,
                "parse_success": parse_success,
                "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                "completion_tokens": usage.get("completion_tokens") if usage else None,
                "total_tokens": usage.get("total_tokens") if usage else None,
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} generation results")
    return df


def check_result_integrity(
    generations_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    config: Optional[Dict] = None,
    allow_incomplete: bool = False,
) -> None:
    """Validate completeness and integrity of generation results.
    
    Checks:
    - max(budgets) <= generations_per_question (always raises)
    - rows == questions * gens
    - no missing or short questions
    - no duplicate (question_id, generation_id)
    - no unexpected IDs (question_id == -1, or not in dataset, or generation_id not in 1..gens)
    - no failed requests (request_ok == False)
    
    Args:
        generations_df: DataFrame of parsed generations.
        dataset_df: DataFrame of dataset questions.
        config: Configuration dictionary.
        allow_incomplete: If True, warn instead of raising IncompleteResultsError.
    
    Raises:
        IncompleteResultsError: If integrity check fails and allow_incomplete is False,
            or if max(budgets) > generations_per_question.
    """
    if config is None:
        config = load_config()
    
    budgets = config["experiment"]["inference_budgets"]
    expected_gens = config["experiment"]["generations_per_question"]
    
    # Always raise if max(budgets) > generations_per_question
    if max(budgets) > expected_gens:
        raise IncompleteResultsError(
            f"Maximum inference budget ({max(budgets)}) exceeds "
            f"generations_per_question ({expected_gens})"
        )
    
    expected_questions = len(dataset_df)
    expected_total_rows = expected_questions * expected_gens
    dataset_qids = set(dataset_df["question_id"].unique())
    
    issues = []
    
    # 1. Total row count check
    if len(generations_df) != expected_total_rows:
        issues.append(
            f"Row count mismatch: expected {expected_total_rows} "
            f"({expected_questions} questions * {expected_gens} gens), got {len(generations_df)}"
        )
    
    # 2. Unexpected IDs (e.g. question_id == -1, or not in dataset_df, or generation_id not in 1..G)
    unexpected = generations_df[
        (~generations_df["question_id"].isin(dataset_qids)) |
        (generations_df["generation_id"] < 1) |
        (generations_df["generation_id"] > expected_gens)
    ]
    if len(unexpected) > 0:
        issues.append(
            f"Found {len(unexpected)} row(s) with unexpected or unparseable custom_ids"
        )
    
    # 3. Duplicate (question_id, generation_id)
    valid_id_rows = generations_df[generations_df["question_id"].isin(dataset_qids)]
    dupes = valid_id_rows.duplicated(subset=["question_id", "generation_id"]).sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate (question_id, generation_id) pairs")
    
    # 4. Missing or short questions
    gen_counts = generations_df.groupby("question_id").size()
    for q_id in dataset_qids:
        cnt = gen_counts.get(q_id, 0)
        if cnt < expected_gens:
            issues.append(
                f"Question {q_id} has {cnt} generations (expected {expected_gens})"
            )
            break
    
    # 5. Failed requests
    if "request_ok" in generations_df.columns:
        failed_count = (~generations_df["request_ok"]).sum()
        if failed_count > 0:
            issues.append(f"Found {failed_count} failed request(s)")
    
    if issues:
        msg = "Result integrity check failed:\n  - " + "\n  - ".join(issues)
        if not allow_incomplete:
            raise IncompleteResultsError(msg)
        else:
            print(f"WARNING: {msg}")


def run_evaluation(
    generations_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    config: Optional[Dict] = None,
    allow_incomplete: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full evaluation across all inference budgets.
    
    For each question:
    1. Retrieve its generations
    2. Sort by generation_id BEFORE the seeded shuffle, so results don't depend on file order
    3. Deterministically shuffle them using seed (shuffle_seed + question_id)
    4. For each N in inference_budgets:
        - Select first N answers
        - Perform majority vote
        - Compare to gold answer
    
    Args:
        generations_df: DataFrame with parsed generations.
        dataset_df: DataFrame with questions and gold answers.
        config: Configuration dictionary.
        allow_incomplete: If True, warn instead of raising error on incomplete results,
            and keep the full denominator (count missing questions as incorrect).
    
    Returns:
        Tuple of (wide_results_df, long_results_df).
    """
    if config is None:
        config = load_config()
    
    check_result_integrity(
        generations_df, dataset_df, config, allow_incomplete=allow_incomplete
    )
    
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
        
        # Sort a question's generations by generation_id BEFORE the seeded shuffle,
        # so results don't depend on results-file order
        q_gens = q_gens.sort_values("generation_id").reset_index(drop=True)
        
        # Deterministically shuffle generations
        rng = random.Random(shuffle_seed + q_id)
        gen_indices = list(range(len(q_gens)))
        rng.shuffle(gen_indices)
        q_gens = q_gens.iloc[gen_indices].reset_index(drop=True)
        
        # Get all answers in shuffled order
        all_answers = q_gens["parsed_answer"].tolist() if len(q_gens) > 0 else []
        
        wide_record = {
            "question_id": q_id,
            "gold_answer": gold,
        }
        
        for n in budgets:
            if len(all_answers) > 0:
                predicted, vote_count, valid_count, vote_dist = evaluate_at_n(
                    all_answers, n
                )
                correct = answers_match(predicted, gold)
            else:
                predicted = None
                vote_count = 0
                valid_count = 0
                correct = False
            
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
