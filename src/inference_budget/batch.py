"""Batch JSONL generation for Sheshnag API."""

import json
import os
import re
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
    
    Creates G requests per question.
    
    Args:
        dataset: DataFrame with questions (must have question_id and question columns).
        model_id: The model identifier.
        config: Configuration dictionary.
        output_path: Where to save the JSONL. Defaults to config path.
    
    Returns:
        Path to the generated JSONL file.
    """
    import copy
    if config is None:
        config = load_config()
    else:
        base_config = load_config()
        merged = copy.deepcopy(base_config)
        for k, v in config.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k].update(v)
            else:
                merged[k] = v
        config = merged
    
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
    - Exactly sample_size * generations_per_question lines
    - Unique custom IDs with format gsm8k_q{q_id:06d}_g{g_id:04d}
    - Exactly sample_size distinct questions
    - Generation IDs exactly 1..G per question
    - Method POST, url /v1/chat/completions
    - Identical generation params on all requests, equal to config params
    - Exactly one prompt per question, with an identical instruction prefix
    - Same model ID for all requests (and not placeholder)
    - No n > 1, no streaming
    - No "sk-..." token or SHESHNAG_API_KEY value in any line (no secret search in text)
    
    Args:
        batch_path: Path to the JSONL file.
        config: Configuration dictionary.
    
    Returns:
        List of error messages. Empty list means validation passed.
    """
    import copy
    if config is None:
        config = load_config()
    else:
        base_config = load_config()
        merged = copy.deepcopy(base_config)
        for k, v in config.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k].update(v)
            else:
                merged[k] = v
        config = merged
    
    errors = []
    custom_ids = set()
    question_gens = {}      # q_id -> set of g_ids
    question_prompts = {}   # q_id -> prompt text
    model_ids = set()
    line_count = 0
    
    expected_sample_size = config["dataset"]["sample_size"]
    expected_gen = config["experiment"]["generations_per_question"]
    expected_total = expected_sample_size * expected_gen
    expected_gen_params = get_generation_params(config)
    
    # Get expected instruction prefix
    template = load_prompt_template(config)
    expected_instruction_prefix = template.split("{question}")[0]
    
    # Check for live SHESHNAG_API_KEY from environment
    env_api_key = os.environ.get("SHESHNAG_API_KEY", "")
    has_env_key = bool(env_api_key and "REPLACE" not in env_api_key and len(env_api_key) > 5)
    
    first_seen_params = None
    
    try:
        with open(batch_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                
                line_count += 1
                
                # Check for "sk-..." token or SHESHNAG_API_KEY in line
                if re.search(r'sk-[a-zA-Z0-9_\-]{8,}', raw_line):
                    errors.append(f"Line {line_num}: Leaked API key token found ('sk-...')")
                if has_env_key and env_api_key in raw_line:
                    errors.append(f"Line {line_num}: Leaked SHESHNAG_API_KEY found in line")
                
                # Parse JSON
                try:
                    request = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check required fields
                for field in ["custom_id", "method", "url", "body"]:
                    if field not in request:
                        errors.append(f"Line {line_num}: Missing field '{field}'")
                
                # Check method and url
                method = request.get("method")
                if method != "POST":
                    errors.append(f"Line {line_num}: Method is '{method}', expected 'POST'")
                
                url = request.get("url")
                if url != "/v1/chat/completions":
                    errors.append(f"Line {line_num}: URL is '{url}', expected '/v1/chat/completions'")
                
                # Check custom_id uniqueness & format
                cid = request.get("custom_id", "")
                if not cid:
                    errors.append(f"Line {line_num}: Missing or empty custom_id")
                    continue
                
                if cid in custom_ids:
                    errors.append(f"Line {line_num}: Duplicate custom_id '{cid}'")
                custom_ids.add(cid)
                
                match = re.match(r'^gsm8k_q(\d+)_g(\d+)$', cid)
                if not match:
                    errors.append(f"Line {line_num}: Invalid custom_id format '{cid}'")
                    continue
                
                q_id = int(match.group(1))
                g_id = int(match.group(2))
                
                if q_id not in question_gens:
                    question_gens[q_id] = set()
                question_gens[q_id].add(g_id)
                
                body = request.get("body", {})
                if not isinstance(body, dict):
                    errors.append(f"Line {line_num}: Request body is not a JSON object")
                    continue
                
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
                
                # Check generation parameters
                actual_params = {}
                for k in expected_gen_params:
                    if k in body:
                        actual_params[k] = body[k]
                
                if actual_params != expected_gen_params:
                    errors.append(
                        f"Line {line_num}: Generation params {actual_params} do not match config {expected_gen_params}"
                    )
                
                if first_seen_params is None:
                    first_seen_params = actual_params
                elif actual_params != first_seen_params:
                    errors.append(
                        f"Line {line_num}: Generation params not identical to previous requests"
                    )
                
                # Check messages and prompt
                messages = body.get("messages", [])
                if not messages or not isinstance(messages, list):
                    errors.append(f"Line {line_num}: Missing or invalid messages list")
                else:
                    user_content = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
                    # Check identical instruction prefix
                    if not user_content.startswith(expected_instruction_prefix):
                        errors.append(
                            f"Line {line_num}: Prompt does not start with expected instruction prefix"
                        )
                    # Check exactly one prompt per question
                    if q_id not in question_prompts:
                        question_prompts[q_id] = user_content
                    elif question_prompts[q_id] != user_content:
                        errors.append(
                            f"Line {line_num}: Question {q_id} has inconsistent prompt across generations"
                        )
    
    except FileNotFoundError:
        errors.append(f"Batch file not found: {batch_path}")
        return errors
    
    # Check total line count
    if line_count != expected_total:
        errors.append(
            f"Expected {expected_total} requests but found {line_count}"
        )
    
    # Check distinct questions
    if len(question_prompts) != expected_sample_size:
        errors.append(
            f"Expected {expected_sample_size} distinct questions, found {len(question_prompts)}"
        )
    
    # Check generation IDs exactly 1..G per question
    expected_gens_set = set(range(1, expected_gen + 1))
    for q_id, g_set in question_gens.items():
        if g_set != expected_gens_set:
            errors.append(
                f"Question {q_id}: generation IDs {sorted(g_set)} != expected 1..{expected_gen}"
            )
    
    # Check model consistency
    if len(model_ids) > 1:
        errors.append(f"Multiple model IDs found: {model_ids}")
    
    # Check placeholder model
    if model_ids and any("REPLACE" in str(m) for m in model_ids):
        errors.append(
            "Model ID still contains placeholder 'REPLACE'. "
            "Update SHESHNAG_MODEL_ID in .env before submission."
        )
    
    return errors
