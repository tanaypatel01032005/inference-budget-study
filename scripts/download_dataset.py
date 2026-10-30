#!/usr/bin/env python3
"""Download and prepare the GSM8K dataset for the inference budget study.

Usage:
    python scripts/download_dataset.py [--force]

Options:
    --force    Force regeneration even if the dataset already exists.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.dataset import (
    download_gsm8k,
    select_questions,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the GSM8K dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if dataset already exists.",
    )
    args = parser.parse_args()
    
    config = load_config()
    output_path = resolve_path(config["paths"]["dataset_processed"])
    
    if output_path.exists() and not args.force:
        print(f"Dataset already exists at {output_path}")
        print("Use --force to regenerate.")
        return
    
    # Download GSM8K
    gsm8k = download_gsm8k()
    test_split = gsm8k["test"]
    
    print(f"GSM8K test split: {len(test_split)} questions")
    
    # Select questions
    sample_size = config["dataset"]["sample_size"]
    random_seed = config["dataset"]["random_seed"]
    
    print(f"Selecting {sample_size} questions with seed {random_seed}...")
    df, selected_indices = select_questions(
        test_split, sample_size=sample_size, random_seed=random_seed
    )
    
    # Validate
    assert len(df) == sample_size, f"Expected {sample_size}, got {len(df)}"
    assert df["question_id"].nunique() == sample_size, "Duplicate question IDs"
    
    # Save
    save_dataset(df, selected_indices, config)
    
    # Update metadata with total source count
    import json
    metadata_path = resolve_path(config["paths"]["dataset_metadata"])
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["total_source_questions"] = len(test_split)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"  Questions selected: {len(df)}")
    print(f"  Output: {output_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
