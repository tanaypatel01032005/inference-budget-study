"""Dataset loading and preparation for GSM8K."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import PROJECT_ROOT, load_config, resolve_path


def download_gsm8k() -> "datasets.DatasetDict":
    """Download the GSM8K dataset using Hugging Face datasets.
    
    Returns:
        The GSM8K dataset dict.
    """
    from datasets import load_dataset
    
    print("Downloading GSM8K dataset from Hugging Face...")
    dataset = load_dataset("gsm8k", "main")
    print(f"Download complete. Splits: {list(dataset.keys())}")
    return dataset


def extract_gold_answer(answer_text: str) -> str:
    """Extract the gold answer from a GSM8K answer string.
    
    GSM8K answers contain reasoning followed by #### <answer>.
    
    Args:
        answer_text: The full answer text from GSM8K.
    
    Returns:
        The extracted numerical answer string.
    
    Raises:
        ValueError: If no answer marker is found.
    """
    if "####" not in answer_text:
        raise ValueError(f"No #### marker found in answer: {answer_text[:100]}...")
    
    # Split on #### and take the last part
    parts = answer_text.split("####")
    raw_answer = parts[-1].strip()
    
    return raw_answer


def select_questions(
    test_split,
    sample_size: int = 1000,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, List[int]]:
    """Randomly select questions from the GSM8K test split.
    
    Args:
        test_split: The test split of GSM8K dataset.
        sample_size: Number of questions to select.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (DataFrame with selected questions, list of selected indices).
    
    Raises:
        ValueError: If test split has fewer questions than sample_size.
    """
    total = len(test_split)
    if total < sample_size:
        raise ValueError(
            f"GSM8K test split has only {total} questions, "
            f"but {sample_size} were requested."
        )
    
    rng = random.Random(random_seed)
    indices = list(range(total))
    selected_indices = sorted(rng.sample(indices, sample_size))
    
    records = []
    for rank, idx in enumerate(selected_indices):
        item = test_split[idx]
        gold_answer = extract_gold_answer(item["answer"])
        records.append({
            "question_id": rank + 1,  # 1-indexed
            "original_index": idx,
            "question": item["question"],
            "full_answer": item["answer"],
            "gold_answer": gold_answer,
        })
    
    df = pd.DataFrame(records)
    return df, selected_indices


def save_dataset(
    df: pd.DataFrame,
    selected_indices: List[int],
    config: Optional[Dict] = None,
) -> Path:
    """Save the selected dataset and metadata.
    
    Args:
        df: DataFrame with selected questions.
        selected_indices: List of original indices.
        config: Configuration dictionary.
    
    Returns:
        Path to the saved dataset.
    """
    if config is None:
        config = load_config()
    
    # Save processed dataset
    output_path = resolve_path(config["paths"]["dataset_processed"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} questions to {output_path}")
    
    # Save metadata
    metadata_path = resolve_path(config["paths"]["dataset_metadata"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    metadata = {
        "source_dataset": "gsm8k",
        "source_config": "main",
        "split": "test",
        "total_source_questions": None,  # Will be set by caller
        "selected_question_count": len(df),
        "random_seed": config["dataset"]["random_seed"],
        "selection_date": datetime.now().isoformat(),
        "selected_indices": selected_indices,
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return output_path


def load_processed_dataset(config: Optional[Dict] = None) -> pd.DataFrame:
    """Load the processed 1,000-question dataset.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        DataFrame with the selected questions.
    
    Raises:
        FileNotFoundError: If the processed dataset does not exist.
    """
    if config is None:
        config = load_config()
    
    dataset_path = resolve_path(config["paths"]["dataset_processed"])
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            f"Run: python scripts/download_dataset.py"
        )
    
    df = pd.read_csv(dataset_path)
    
    expected = config["dataset"]["sample_size"]
    actual = len(df)
    if actual != expected:
        raise ValueError(
            f"Expected {expected} questions but found {actual} in {dataset_path}"
        )
    
    return df
