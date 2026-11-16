#!/usr/bin/env python3
"""Submit the batch to Sheshnag and optionally poll for completion.

Usage:
    python scripts/submit_sheshnag.py [--no-poll] [--poll-interval 30]
    python scripts/submit_sheshnag.py --check-models
    python scripts/submit_sheshnag.py --status BATCH_ID
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.config import load_config, load_env, get_sheshnag_config, resolve_path
from src.inference_budget.batch import validate_batch_file
from src.inference_budget.sheshnag import (
    create_client,
    list_models,
    verify_model,
    upload_batch_file,
    create_batch,
    poll_batch_status,
    download_results,
)


def cmd_check_models():
    """List available models on Sheshnag."""
    print("Querying available models...")
    models = list_models()
    print(f"\nAvailable models ({len(models)}):")
    for m in models:
        print(f"  - {m.id}")


def cmd_submit(args):
    """Submit the batch."""
    config = load_config()
    sheshnag_config = get_sheshnag_config()
    
    batch_path = resolve_path(config["paths"]["batch_output"])
    if not batch_path.exists():
        print(f"ERROR: Batch file not found: {batch_path}")
        print("Run: python scripts/create_batch.py")
        sys.exit(1)
    
    # Verify model
    model_id = sheshnag_config["model_id"]
    print(f"Verifying model '{model_id}'...")
    if not verify_model(model_id):
        print("ERROR: Model not available. Run --check-models to see available models.")
        sys.exit(1)
    
    # Validate batch file before upload
    print(f"Validating batch file: {batch_path}")
    errors = validate_batch_file(batch_path, config)
    if errors:
        print(f"ERROR: Batch validation failed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    # Abort if the model in the batch file's first line != SHESHNAG_MODEL_ID
    with open(batch_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        print("ERROR: Batch file is empty.")
        sys.exit(1)
    try:
        first_req = json.loads(first_line)
        batch_model = first_req.get("body", {}).get("model")
    except Exception as e:
        print(f"ERROR: Could not parse first line of batch file: {e}")
        sys.exit(1)
    
    if batch_model != model_id:
        print(
            f"ERROR: Model mismatch! Model in batch file ('{batch_model}') "
            f"!= SHESHNAG_MODEL_ID ('{model_id}')."
        )
        sys.exit(1)
    
    # Upload
    print(f"\nUploading batch file: {batch_path}")
    file_id = upload_batch_file(batch_path)
    
    # Create batch
    print(f"\nCreating batch job...")
    batch_id = create_batch(file_id)
    
    # Save batch ID
    batch_info_path = resolve_path("batch/results/batch_info.json")
    batch_info_path.parent.mkdir(parents=True, exist_ok=True)
    batch_info = {
        "batch_id": batch_id,
        "file_id": file_id,
        "model_id": model_id,
        "batch_file": str(batch_path),
    }
    with open(batch_info_path, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2)
    print(f"\nBatch info saved to: {batch_info_path}")
    print(f"BATCH ID: {batch_id}")
    
    if args.no_poll:
        print("\nNot polling. Check status manually:")
        print(f"  python scripts/submit_sheshnag.py --status {batch_id}")
        return
    
    # Poll
    print(f"\nPolling for completion (interval: {args.poll_interval}s)...")
    result = poll_batch_status(batch_id, poll_interval=args.poll_interval)
    
    print(f"\nBatch completed with status: {result['status']}")
    
    if result["status"] == "completed":
        print("Downloading results...")
        output_path = download_results(batch_id, config=config)
        print(f"Results saved to: {output_path}")
        print("\nNext step: python scripts/evaluate.py")
    else:
        print(f"Batch ended with status: {result['status']}")
        if result.get("error_file_id"):
            print("Error details may be available. Check batch info.")


def cmd_status(batch_id):
    """Check batch status."""
    client = create_client()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch ID: {batch.id}")
    print(f"Status:   {batch.status}")
    if batch.request_counts:
        print(f"Total:    {batch.request_counts.total}")
        print(f"Completed: {batch.request_counts.completed}")
        print(f"Failed:    {batch.request_counts.failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch to Sheshnag and manage jobs."
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Check status of a batch by ID.",
    )
    parser.add_argument(
        "--no-poll",
        action="store_true",
        help="Submit without polling for completion.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls (default: 30).",
    )
    args = parser.parse_args()
    
    load_env()
    
    if args.check_models:
        get_sheshnag_config()  # Validate config first
        cmd_check_models()
    elif args.status:
        get_sheshnag_config()
        cmd_status(args.status)
    else:
        cmd_submit(args)


if __name__ == "__main__":
    main()
