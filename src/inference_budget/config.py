"""Configuration management for the inference budget study."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the YAML configuration file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml.
    
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def load_env() -> None:
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    
    if load_dotenv is not None:
        load_dotenv(env_path)
    else:
        # Fallback simple .env parser
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val


def get_sheshnag_config() -> Dict[str, str]:
    """Get Sheshnag configuration from environment variables.
    
    Returns:
        Dictionary with base_url, api_key, and model_id.
    
    Raises:
        ValueError: If any required value is missing or still a placeholder.
    """
    load_env()
    
    base_url = os.environ.get("SHESHNAG_BASE_URL", "")
    api_key = os.environ.get("SHESHNAG_API_KEY", "")
    model_id = os.environ.get("SHESHNAG_MODEL_ID", "")
    
    errors = []
    
    if not base_url or "REPLACE" in base_url:
        errors.append(
            "SHESHNAG_BASE_URL is not configured. "
            "Set it in .env (copy from .env.example)."
        )
    if not api_key or "REPLACE" in api_key:
        errors.append(
            "SHESHNAG_API_KEY is not configured. "
            "Set it in .env (copy from .env.example)."
        )
    if not model_id or "REPLACE" in model_id:
        errors.append(
            "SHESHNAG_MODEL_ID is not configured. "
            "Set it in .env. Obtain the model ID from GET /v1/models on your Sheshnag deployment."
        )
    
    if errors:
        raise ValueError(
            "Sheshnag configuration incomplete:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model_id": model_id,
    }


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path from the project root.
    
    Args:
        relative_path: Path relative to project root.
    
    Returns:
        Absolute path.
    """
    return PROJECT_ROOT / relative_path


def get_generation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract generation parameters from config.
    
    Args:
        config: Full configuration dictionary.
    
    Returns:
        Dictionary of generation parameters.
    """
    gen = config["generation"]
    params = {
        "temperature": gen["temperature"],
        "top_p": gen["top_p"],
        "max_tokens": gen["max_tokens"],
    }
    # Only include seed if it is not null
    if gen.get("seed") is not None:
        params["seed"] = gen["seed"]
    return params
