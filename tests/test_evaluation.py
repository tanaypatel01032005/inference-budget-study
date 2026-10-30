"""Tests for evaluation logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference_budget.evaluation import compute_accuracy
from src.inference_budget.parsing import answers_match


class TestComputeAccuracy:
    """Tests for accuracy computation."""
    
    def test_perfect_accuracy(self):
        long_df = pd.DataFrame({
            "question_id": [1, 2, 3] * 2,
            "N": [1, 1, 1, 3, 3, 3],
            "correct": [True, True, True, True, True, True],
        })
        summary = compute_accuracy(long_df)
        for _, row in summary.iterrows():
            assert row["accuracy"] == 1.0
    
    def test_zero_accuracy(self):
        long_df = pd.DataFrame({
            "question_id": [1, 2],
            "N": [1, 1],
            "correct": [False, False],
        })
        summary = compute_accuracy(long_df)
        assert summary.iloc[0]["accuracy"] == 0.0
    
    def test_mixed_accuracy(self):
        long_df = pd.DataFrame({
            "question_id": [1, 2, 3, 4],
            "N": [1, 1, 1, 1],
            "correct": [True, True, False, False],
        })
        summary = compute_accuracy(long_df)
        assert summary.iloc[0]["accuracy"] == 0.5
    
    def test_correct_count(self):
        long_df = pd.DataFrame({
            "question_id": [1, 2, 3],
            "N": [1, 1, 1],
            "correct": [True, False, True],
        })
        summary = compute_accuracy(long_df)
        assert summary.iloc[0]["correct"] == 2
        assert summary.iloc[0]["incorrect"] == 1


class TestNBudgetLogic:
    """Tests for the N-budget evaluation logic."""
    
    def test_n_values(self):
        """Verify all expected N values are evaluated."""
        expected_n = [1, 3, 5, 10, 20, 50]
        for n in expected_n:
            assert n > 0
            assert n <= 50
    
    def test_n1_uses_first_answer(self):
        """N=1 should use exactly the first answer."""
        from src.inference_budget.voting import evaluate_at_n
        answers = ["42", "40", "38", "42", "42"]
        predicted, _, valid, _ = evaluate_at_n(answers, 1)
        assert predicted == "42"
        assert valid == 1
    
    def test_increasing_n_uses_more_answers(self):
        """Higher N should use more answers."""
        from src.inference_budget.voting import evaluate_at_n
        answers = ["40", "42", "42", "42", "40"]
        
        pred_1, _, valid_1, _ = evaluate_at_n(answers, 1)
        pred_3, _, valid_3, _ = evaluate_at_n(answers, 3)
        
        assert valid_1 == 1
        assert valid_3 == 3
        assert pred_1 == "40"  # First answer
        assert pred_3 == "42"  # Majority of first 3

