"""Tests for dataset utilities."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.dataset import extract_gold_answer


class TestExtractGoldAnswer:
    """Tests for GSM8K gold answer extraction."""
    
    def test_simple(self):
        assert extract_gold_answer("reasoning\n#### 42") == "42"
    
    def test_with_spaces(self):
        assert extract_gold_answer("reasoning\n####  42 ") == "42"
    
    def test_negative(self):
        assert extract_gold_answer("reasoning\n#### -5") == "-5"
    
    def test_decimal(self):
        assert extract_gold_answer("reasoning\n#### 3.5") == "3.5"
    
    def test_comma_number(self):
        assert extract_gold_answer("reasoning\n#### 1,200") == "1,200"
    
    def test_multi_step(self):
        text = "Step 1: ...\nStep 2: ...\n#### 100"
        assert extract_gold_answer(text) == "100"
    
    def test_no_marker(self):
        with pytest.raises(ValueError):
            extract_gold_answer("no marker here")
    
    def test_multiple_markers(self):
        """Should use the last #### marker."""
        text = "#### wrong\nmore reasoning\n#### 42"
        assert extract_gold_answer(text) == "42"

