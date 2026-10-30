#!/usr/bin/env python3
"""Validate the Sheshnag batch JSONL file.

Usage:
    python scripts/validate_batch.py [--batch-file PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.batch import validate_batch_file


def main():
    parser = argparse.ArgumentParser(
        description="Validate the Sheshnag batch JSONL file."
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Path to the batch JSONL file.",
    )
    args = parser.parse_args()
    
    config = load_config()
    
    if args.batch_file:
        batch_path = Path(args.batch_file)
    else:
        batch_path = resolve_path(config["paths"]["batch_output"])
    
    if not batch_path.exists():
        print(f"ERROR: Batch file not found: {batch_path}")
        print("Run: python scripts/create_batch.py")
        sys.exit(1)
    
    print(f"Validating: {batch_path}")
    print()
    
    errors = validate_batch_file(batch_path, config)
    
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)
    else:
        # Print summary
        import json
        line_count = 0
        with open(batch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        
        expected = (
            config["dataset"]["sample_size"]
            * config["experiment"]["generations_per_question"]
        )
        
        print(f"VALIDATION PASSED")
        print(f"  Total requests: {line_count}")
        print(f"  Expected:       {expected}")
        print(f"  Questions:      {config['dataset']['sample_size']}")
        print(f"  Gens/question:  {config['experiment']['generations_per_question']}")
        print(f"\nBatch is ready for submission.")


if __name__ == "__main__":
    main()
