"""Sheshnag API client for batch submission and result retrieval."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_sheshnag_config, load_config, resolve_path


def create_client():
    """Create an OpenAI-compatible client for Sheshnag.
    
    Returns:
        openai.OpenAI client configured for Sheshnag.
    """
    from openai import OpenAI
    
    config = get_sheshnag_config()
    
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
    )
    
    return client


def list_models() -> list:
    """List available models on the Sheshnag deployment.
    
    Returns:
        List of model objects.
    """
    client = create_client()
    models = client.models.list()
    return list(models)


def verify_model(model_id: str) -> bool:
    """Verify that a model is available on Sheshnag.
    
    Args:
        model_id: The model ID to check.
    
    Returns:
        True if the model is available.
    """
    models = list_models()
    available_ids = [m.id for m in models]
    
    if model_id not in available_ids:
        print(f"Model '{model_id}' not found.")
        print(f"Available models: {available_ids}")
        return False
    
    print(f"Model '{model_id}' is available.")
    return True


def upload_batch_file(batch_path: Path) -> str:
    """Upload a batch JSONL file to Sheshnag.
    
    Args:
        batch_path: Path to the JSONL file.
    
    Returns:
        The uploaded file ID.
    """
    client = create_client()
    
    with open(batch_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    
    print(f"Uploaded batch file. File ID: {uploaded.id}")
    return uploaded.id


def create_batch(file_id: str) -> str:
    """Create a batch job on Sheshnag.
    
    Args:
        file_id: The uploaded file ID.
    
    Returns:
        The batch ID.
    """
    client = create_client()
    
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    
    print(f"Created batch. Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    return batch.id


def poll_batch_status(
    batch_id: str,
    poll_interval: int = 30,
    max_wait: int = 86400,
) -> Dict[str, Any]:
    """Poll batch status until completion or failure.
    
    Args:
        batch_id: The batch ID to monitor.
        poll_interval: Seconds between status checks.
        max_wait: Maximum seconds to wait.
    
    Returns:
        Final batch status object as dict.
    """
    client = create_client()
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    
    elapsed = 0
    while elapsed < max_wait:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        print(
            f"[{elapsed}s] Batch {batch_id}: {status} "
            f"(completed: {batch.request_counts.completed if batch.request_counts else '?'} / "
            f"total: {batch.request_counts.total if batch.request_counts else '?'})"
        )
        
        if status in terminal_statuses:
            return {
                "id": batch.id,
                "status": batch.status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_counts": {
                    "total": batch.request_counts.total if batch.request_counts else 0,
                    "completed": batch.request_counts.completed if batch.request_counts else 0,
                    "failed": batch.request_counts.failed if batch.request_counts else 0,
                },
            }
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    raise TimeoutError(
        f"Batch {batch_id} did not complete within {max_wait}s"
    )


def download_results(
    batch_id: str,
    output_path: Optional[Path] = None,
    config: Optional[Dict] = None,
) -> Path:
    """Download batch results to a local file.
    
    Args:
        batch_id: The batch ID.
        output_path: Where to save results. Defaults to config path.
        config: Configuration dictionary.
    
    Returns:
        Path to the downloaded results file.
    """
    if config is None:
        config = load_config()
    
    if output_path is None:
        output_path = resolve_path(config["paths"]["results_raw"])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    client = create_client()
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch_id} status is '{batch.status}', not 'completed'"
        )
    
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output file ID")
    
    # Download the output file
    content = client.files.content(batch.output_file_id)
    
    with open(output_path, "wb") as f:
        f.write(content.read())
    
    print(f"Downloaded results to {output_path}")
    
    # Also download error file if it exists
    if batch.error_file_id:
        error_path = output_path.parent / "sheshnag_errors.jsonl"
        error_content = client.files.content(batch.error_file_id)
        with open(error_path, "wb") as f:
            f.write(error_content.read())
        print(f"Downloaded error file to {error_path}")
    
    return output_path
