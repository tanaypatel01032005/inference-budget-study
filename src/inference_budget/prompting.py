"""Prompt template management."""

from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT, load_config, resolve_path


def load_prompt_template(config: Optional[dict] = None) -> str:
    """Load the prompt template from file.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        The prompt template string with {question} placeholder.
    
    Raises:
        FileNotFoundError: If the prompt template file does not exist.
    """
    if config is None:
        config = load_config()
    
    template_path = resolve_path(config["paths"]["prompt_template"])
    
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found at {template_path}"
        )
    
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()
    
    # Validate that the template contains the question placeholder
    if "{question}" not in template:
        raise ValueError(
            "Prompt template must contain {question} placeholder"
        )
    
    return template


def format_prompt(template: str, question: str) -> str:
    """Format a prompt with the given question.
    
    Args:
        template: Prompt template with {question} placeholder.
        question: The math problem text.
    
    Returns:
        The formatted prompt string.
    """
    return template.format(question=question)
