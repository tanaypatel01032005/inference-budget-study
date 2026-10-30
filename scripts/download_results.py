#!/usr/bin/env python3
"""Download results from a completed Sheshnag batch.

Usage:
    python scripts/download_results.py --batch-id BATCH_ID
    python scripts/download_results.py  # uses saved batch_info.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, load_env, get_sheshnag_config, resolve_path
from src.inference_budget.sheshnag import download_results


def main():
    parser = argparse.ArgumentParser(
        description="Download results from a completed Sheshnag batch."
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID to download. If omitted, reads from batch_info.json.",
    )
    args = parser.parse_args()
    
    config = load_config()
    load_env()
    get_sheshnag_config()  # Validate
    
    batch_id = args.batch_id
    
    if not batch_id:
        batch_info_path = resolve_path("batch/results/batch_info.json")
        if not batch_info_path.exists():
            print("ERROR: No batch ID provided and no batch_info.json found.")
            print("Use --batch-id BATCH_ID or run submit_sheshnag.py first.")
            sys.exit(1)
        with open(batch_info_path, "r", encoding="utf-8") as f:
            batch_info = json.load(f)
        batch_id = batch_info["batch_id"]
        print(f"Using batch ID from batch_info.json: {batch_id}")
    
    output_path = resolve_path(config["paths"]["results_raw"])
    
    if output_path.exists():
        print(f"WARNING: Results file already exists at {output_path}")
        print("Raw results will NOT be overwritten.")
        backup = output_path.with_suffix(f".backup.jsonl")
        print(f"Saving to backup: {backup}")
        output_path = backup
    
    print(f"Downloading results for batch {batch_id}...")
    result_path = download_results(batch_id, output_path=output_path, config=config)
    print(f"\nResults saved to: {result_path}")
    print("\nNext step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
