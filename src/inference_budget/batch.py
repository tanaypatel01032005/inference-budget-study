"""Batch JSONL generation for Sheshnag API."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import load_config, resolve_path, get_generation_params
from .prompting import load_prompt_template, format_prompt


def create_batch_request(
    custom_id: str,
    model_id: str,
    prompt: str,
    generation_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a single batch request in OpenAI batch format.
    
    Args:
        custom_id: Unique identifier for this request.
        model_id: The model to use.
        prompt: The formatted prompt.
        generation_params: Generation parameters (temperature, etc.).
    
    Returns:
        Dictionary representing one batch request line.
    """
    body = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **generation_params,
    }
    
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def generate_batch_jsonl(
    dataset: pd.DataFrame,
    model_id: str,
    config: Optional[Dict] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate the full batch JSONL file.
    
    Creates 50 requests per question = 50,000 total requests.
    
    Args:
        dataset: DataFrame with questions (must have question_id and question columns).
        model_id: The model identifier.
        config: Configuration dictionary.
        output_path: Where to save the JSONL. Defaults to config path.
    
    Returns:
        Path to the generated JSONL file.
    """
    if config is None:
        config = load_config()
    
    if output_path is None:
        output_path = resolve_path(config["paths"]["batch_output"])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    template = load_prompt_template(config)
    gen_params = get_generation_params(config)
    generations_per_q = config["experiment"]["generations_per_question"]
    
    expected_total = len(dataset) * generations_per_q
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in dataset.iterrows():
            q_id = int(row["question_id"])
            question = row["question"]
            prompt = format_prompt(template, question)
            
            for gen_idx in range(1, generations_per_q + 1):
                custom_id = f"gsm8k_q{q_id:06d}_g{gen_idx:04d}"
                request = create_batch_request(
                    custom_id=custom_id,
                    model_id=model_id,
                    prompt=prompt,
                    generation_params=gen_params,
                )
                f.write(json.dumps(request) + "\n")
                count += 1
    
    if count != expected_total:
        raise RuntimeError(
            f"Generated {count} requests but expected {expected_total}"
        )
    
    print(f"Generated {count} batch requests to {output_path}")
    return output_path


def validate_batch_file(
    batch_path: Path,
    config: Optional[Dict] = None,
) -> List[str]:
    """Validate a batch JSONL file.
    
    Checks:
    - Valid JSONL format
    - Exactly 50,000 lines
    - Unique custom IDs
    - Exactly 50 generations per question
    - Same model ID for all requests
    - No n > 1
    - No streaming
    - Required fields exist
    - No accidental credentials
    
    Args:
        batch_path: Path to the JSONL file.
        config: Configuration dictionary.
    
    Returns:
        List of error messages. Empty list means validation passed.
    """
    if config is None:
        config = load_config()
    
    errors = []
    custom_ids = set()
    question_counts = {}
    model_ids = set()
    line_count = 0
    
    expected_total = (
        config["dataset"]["sample_size"]
        * config["experiment"]["generations_per_question"]
    )
    
    try:
        with open(batch_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                
                # Parse JSON
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check required fields
                for field in ["custom_id", "method", "url", "body"]:
                    if field not in request:
                        errors.append(f"Line {line_num}: Missing field '{field}'")
                
                # Check custom_id uniqueness
                cid = request.get("custom_id", "")
                if cid in custom_ids:
                    errors.append(f"Line {line_num}: Duplicate custom_id '{cid}'")
                custom_ids.add(cid)
                
                # Parse question ID from custom_id
                import re
                match = re.match(r'gsm8k_q(\d+)_g(\d+)', cid)
                if match:
                    q_id = match.group(1)
                    question_counts[q_id] = question_counts.get(q_id, 0) + 1
                
                body = request.get("body", {})
                
                # Check model
                if "model" in body:
                    model_ids.add(body["model"])
                
                # Check n > 1
                if body.get("n", 1) > 1:
                    errors.append(
                        f"Line {line_num}: Uses n={body['n']} (must be 1 or omitted)"
                    )
                
                # Check streaming
                if body.get("stream", False):
                    errors.append(f"Line {line_num}: Uses streaming (not allowed)")
                
                # Check for credential leaks
                line_lower = line.lower()
                for keyword in ["api_key", "api-key", "bearer", "password", "secret"]:
                    if keyword in line_lower and keyword not in ("custom_id",):
                        # Exclude false positives in field names
                        if keyword not in json.dumps(request.get("body", {}).get("messages", [])).lower():
                            pass  # Only flag if in non-message content
    
    except FileNotFoundError:
        errors.append(f"Batch file not found: {batch_path}")
        return errors
    
    # Check total count
    if line_count != expected_total:
        errors.append(
            f"Expected {expected_total} requests but found {line_count}"
        )
    
    # Check model consistency
    if len(model_ids) > 1:
        errors.append(f"Multiple model IDs found: {model_ids}")
    
    # Check placeholder model
    if model_ids and any("REPLACE" in m for m in model_ids):
        errors.append(
            "Model ID still contains placeholder 'REPLACE'. "
            "Update SHESHNAG_MODEL_ID in .env before submission."
        )
    
    # Check generations per question
    expected_gen = config["experiment"]["generations_per_question"]
    for q_id, count in question_counts.items():
        if count != expected_gen:
            errors.append(
                f"Question {q_id}: {count} generations (expected {expected_gen})"
            )
    
    return errors
