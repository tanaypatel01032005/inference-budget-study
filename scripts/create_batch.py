#!/usr/bin/env python3
"""Create the Sheshnag batch JSONL file.

Usage:
    python scripts/create_batch.py [--model-id MODEL_ID]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, load_env, resolve_path
from src.inference_budget.dataset import load_processed_dataset
from src.inference_budget.batch import generate_batch_jsonl


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
    
    # Generate batch
    output_path = resolve_path(config["paths"]["batch_output"])
    batch_path = generate_batch_jsonl(
        dataset=dataset,
        model_id=model_id,
        config=config,
        output_path=output_path,
    )
    
    expected = len(dataset) * config["experiment"]["generations_per_question"]
    print(f"\nBatch file created: {batch_path}")
    print(f"Expected requests: {expected}")
    print(f"\nNext step: python scripts/validate_batch.py")


if __name__ == "__main__":
    main()
