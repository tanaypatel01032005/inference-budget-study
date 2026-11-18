#!/usr/bin/env python3
"""Create the Sheshnag batch JSONL file.

Usage:
    python scripts/create_batch.py [--model-id MODEL_ID]
    python scripts/create_batch.py --questions 5 --generations 5   # pilot
"""

import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, load_env, resolve_path
from src.inference_budget.dataset import load_processed_dataset
from src.inference_budget.batch import generate_batch_jsonl, validate_batch_file

PILOT_BATCH_PATH = "batch/generated/pilot_batch.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Create the Sheshnag batch JSONL file."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model ID (otherwise uses SHESHNAG_MODEL_ID env var).",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Pilot: use only the first Q questions (default: config sample_size).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Pilot: use G generations per question (default: config value).",
    )
    args = parser.parse_args()
    
    config = load_config()
    load_env()
    
    # Determine model ID
    model_id = args.model_id or os.environ.get("SHESHNAG_MODEL_ID", "")
    if not model_id or "REPLACE" in model_id:
        print("ERROR: Model ID not configured.")
        print("Either:")
        print("  1. Set SHESHNAG_MODEL_ID in .env")
        print("  2. Pass --model-id on the command line")
        print()
        print("Get the model ID from: GET /v1/models on your Sheshnag deployment.")
        sys.exit(1)
    
    print(f"Model ID: {model_id}")
    
    # Load dataset
    dataset = load_processed_dataset(config)
    print(f"Loaded {len(dataset)} questions")
    
    pilot = args.questions is not None or args.generations is not None
    if pilot:
        n_q = args.questions if args.questions is not None else config["dataset"]["sample_size"]
        n_g = (args.generations if args.generations is not None
               else config["experiment"]["generations_per_question"])
        if not (1 <= n_q <= len(dataset)) or n_g < 1:
            print(f"ERROR: invalid pilot size: questions={n_q}, generations={n_g}")
            sys.exit(1)
        dataset = dataset.head(n_q)
        # Config copy with matching sizes; the real config is untouched.
        config = copy.deepcopy(config)
        config["dataset"]["sample_size"] = n_q
        config["experiment"]["generations_per_question"] = n_g
        output_path = resolve_path(PILOT_BATCH_PATH)
        print(f"PILOT: {n_q} questions x {n_g} generations")
    else:
        output_path = resolve_path(config["paths"]["batch_output"])
    
    # Generate batch
    batch_path = generate_batch_jsonl(
        dataset=dataset,
        model_id=model_id,
        config=config,
        output_path=output_path,
    )
    
    if pilot:
        errors = validate_batch_file(batch_path, config)
        if errors:
            print(f"ERROR: Pilot batch validation failed with {len(errors)} error(s):")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        print("Pilot batch validation PASSED")
    
    expected = len(dataset) * config["experiment"]["generations_per_question"]
    print(f"\nBatch file created: {batch_path}")
    print(f"Expected requests: {expected}")
    print(f"\nNext step: python scripts/validate_batch.py")


if __name__ == "__main__":
    main()
