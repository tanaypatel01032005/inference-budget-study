"""Tests for answer extraction and normalization."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.parsing import normalize_answer, extract_model_answer, answers_match


class TestNormalizeAnswer:
    """Tests for answer normalization."""
    
    def test_integer(self):
        assert normalize_answer("42") == "42"
    
    def test_float_integer(self):
        assert normalize_answer("42.0") == "42"
    
    def test_float_trailing_zeros(self):
        assert normalize_answer("42.00") == "42"
    
    def test_actual_float(self):
        assert normalize_answer("3.5") == "3.5"
    
    def test_negative(self):
        assert normalize_answer("-5") == "-5"
    
    def test_comma_number(self):
        assert normalize_answer("1,200") == "1200"
    
    def test_large_comma_number(self):
        assert normalize_answer("1,000,000") == "1000000"
    
    def test_dollar_sign(self):
        assert normalize_answer("$42") == "42"
    
    def test_dollar_with_comma(self):
        assert normalize_answer("$1,200") == "1200"
    
    def test_trailing_unit_dollars(self):
        assert normalize_answer("42 dollars") == "42"
    
    def test_trailing_unit_hours(self):
        assert normalize_answer("5 hours") == "5"
    
    def test_percentage(self):
        assert normalize_answer("42%") == "42"
    
    def test_whitespace(self):
        assert normalize_answer("  42  ") == "42"
    
    def test_empty(self):
        assert normalize_answer("") is None
    
    def test_none(self):
        assert normalize_answer(None) is None
    
    def test_different_values_stay_different(self):
        assert normalize_answer("42") != normalize_answer("43")
    
    def test_equivalent_formats(self):
        assert normalize_answer("42") == normalize_answer("42.0")
        assert normalize_answer("1200") == normalize_answer("1,200")
        assert normalize_answer("$42") == normalize_answer("42 dollars")


class TestExtractModelAnswer:
    """Tests for model answer extraction."""
    
    def test_final_answer_format(self):
        response = "Some reasoning...\nFINAL_ANSWER: 42"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "FINAL_ANSWER"
    
    def test_final_answer_with_space(self):
        response = "Reasoning...\nFINAL ANSWER: 42"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "FINAL_ANSWER_SPACE"
    
    def test_hash_format(self):
        response = "Reasoning...\n#### 42"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "HASH_MARKER"
    
    def test_boxed_format(self):
        response = "Reasoning...\n\\boxed{42}"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "BOXED"
    
    def test_unparseable(self):
        response = "I don't know the answer."
        answer, method = extract_model_answer(response)
        assert answer is None
        assert method == "unparseable"
    
    def test_empty_response(self):
        answer, method = extract_model_answer("")
        assert answer is None
        assert method == "unparseable"
    
    def test_none_response(self):
        answer, method = extract_model_answer(None)
        assert answer is None
        assert method == "unparseable"
    
    def test_case_insensitive_final_answer(self):
        response = "Reasoning...\nfinal_answer: 42"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "FINAL_ANSWER"
    
    def test_final_answer_with_comma(self):
        response = "FINAL_ANSWER: 1,200"
        answer, method = extract_model_answer(response)
        assert answer == "1200"
    
    def test_priority_order(self):
        """FINAL_ANSWER should be preferred over #### when both exist."""
        response = "#### 40\nFINAL_ANSWER: 42"
        answer, method = extract_model_answer(response)
        assert answer == "42"
        assert method == "FINAL_ANSWER"


class TestAnswersMatch:
    """Tests for answer comparison."""
    
    def test_same_integer(self):
        assert answers_match("42", "42")
    
    def test_integer_vs_float(self):
        assert answers_match("42", "42.0")
    
    def test_different_values(self):
        assert not answers_match("42", "43")
    
    def test_comma_format(self):
        assert answers_match("1200", "1,200")
    
    def test_none_left(self):
        assert not answers_match(None, "42")
    
    def test_none_right(self):
        assert not answers_match("42", None)
    
    def test_both_none(self):
        assert not answers_match(None, None)

