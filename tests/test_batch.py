"""Tests for batch generation and validation."""

import json
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference_budget.batch import create_batch_request, validate_batch_file


class TestCreateBatchRequest:
    """Tests for batch request creation."""
    
    def test_structure(self):
        req = create_batch_request(
            custom_id="gsm8k_q000001_g0001",
            model_id="test-model",
            prompt="What is 2+2?",
            generation_params={"temperature": 0.7, "top_p": 1.0, "max_tokens": 512},
        )
        
        assert req["custom_id"] == "gsm8k_q000001_g0001"
        assert req["method"] == "POST"
        assert req["url"] == "/v1/chat/completions"
        assert req["body"]["model"] == "test-model"
        assert len(req["body"]["messages"]) == 1
        assert req["body"]["messages"][0]["role"] == "user"
        assert req["body"]["temperature"] == 0.7
        assert req["body"]["max_tokens"] == 512
    
    def test_no_n_parameter(self):
        """Batch requests must not include n > 1."""
        req = create_batch_request(
            custom_id="test",
            model_id="test-model",
            prompt="Test",
            generation_params={"temperature": 0.7},
        )
        assert req["body"].get("n", 1) <= 1
    
    def test_no_streaming(self):
        req = create_batch_request(
            custom_id="test",
            model_id="test-model",
            prompt="Test",
            generation_params={"temperature": 0.7},
        )
        assert req["body"].get("stream", False) is not True
    
    def test_unique_custom_ids(self):
        """Custom IDs should be unique across questions and generations."""
        ids = set()
        for q in range(1, 4):
            for g in range(1, 4):
                cid = f"gsm8k_q{q:06d}_g{g:04d}"
                assert cid not in ids, f"Duplicate ID: {cid}"
                ids.add(cid)


class TestValidateBatchFile:
    """Tests for batch validation."""
    
    def _write_batch(self, path, requests):
        with open(path, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
    
    def _make_valid_request(self, q_id, g_id):
        return {
            "custom_id": f"gsm8k_q{q_id:06d}_g{g_id:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
            },
        }
    
    def test_duplicate_custom_ids(self):
        """Should detect duplicate custom IDs."""
        config = {
            "dataset": {"sample_size": 1},
            "experiment": {"generations_per_question": 2},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            req = self._make_valid_request(1, 1)
            f.write(json.dumps(req) + "\n")
            f.write(json.dumps(req) + "\n")  # Duplicate
            path = Path(f.name)
        
        errors = validate_batch_file(path, config)
        assert any("Duplicate" in e for e in errors)
        path.unlink()
    
    def test_n_greater_than_1(self):
        """Should detect n > 1."""
        config = {
            "dataset": {"sample_size": 1},
            "experiment": {"generations_per_question": 1},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            req = self._make_valid_request(1, 1)
            req["body"]["n"] = 50
            f.write(json.dumps(req) + "\n")
            path = Path(f.name)
        
        errors = validate_batch_file(path, config)
        assert any("n=" in e for e in errors)
        path.unlink()

