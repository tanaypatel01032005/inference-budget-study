"""Tests for majority voting."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_budget.voting import majority_vote, evaluate_at_n


class TestMajorityVote:
    """Tests for majority/plurality voting."""
    
    def test_clear_majority(self):
        answers = ["42", "42", "40", "42", "38"]
        winner, count, dist = majority_vote(answers)
        assert winner == "42"
        assert count == 3
    
    def test_unanimous(self):
        answers = ["42", "42", "42"]
        winner, count, dist = majority_vote(answers)
        assert winner == "42"
        assert count == 3
    
    def test_single_answer(self):
        answers = ["42"]
        winner, count, dist = majority_vote(answers)
        assert winner == "42"
        assert count == 1
    
    def test_tie_breaking_first_occurrence(self):
        """On tie, the answer with earliest first occurrence wins."""
        answers = ["42", "43", "42", "43"]
        winner, count, dist = majority_vote(answers)
        assert winner == "42"
        assert count == 2
    
    def test_tie_breaking_reversed(self):
        """On tie, earliest first occurrence wins (opposite order)."""
        answers = ["43", "42", "43", "42"]
        winner, count, dist = majority_vote(answers)
        assert winner == "43"
        assert count == 2
    
    def test_all_none(self):
        answers = [None, None, None]
        winner, count, dist = majority_vote(answers)
        assert winner is None
        assert count == 0
    
    def test_empty_list(self):
        answers = []
        winner, count, dist = majority_vote(answers)
        assert winner is None
        assert count == 0
    
    def test_some_none(self):
        answers = [None, "42", None, "42", None]
        winner, count, dist = majority_vote(answers)
        assert winner == "42"
        assert count == 2
    
    def test_vote_distribution(self):
        answers = ["42", "40", "42", "40", "42"]
        winner, count, dist = majority_vote(answers)
        assert dist["42"] == 3
        assert dist["40"] == 2


class TestEvaluateAtN:
    """Tests for evaluate_at_n."""
    
    def test_n1(self):
        answers = ["42", "40", "42", "40", "42"]
        predicted, vote_count, valid, dist = evaluate_at_n(answers, 1)
        assert predicted == "42"
        assert valid == 1
    
    def test_n3(self):
        answers = ["42", "40", "42", "40", "42"]
        predicted, vote_count, valid, dist = evaluate_at_n(answers, 3)
        assert predicted == "42"
        assert valid == 3
    
    def test_n5(self):
        answers = ["42", "40", "42", "40", "42"]
        predicted, vote_count, valid, dist = evaluate_at_n(answers, 5)
        assert predicted == "42"
        assert vote_count == 3
        assert valid == 5
    
    def test_n_with_nones(self):
        answers = [None, "42", None, "42", None]
        predicted, vote_count, valid, dist = evaluate_at_n(answers, 5)
        assert predicted == "42"
        assert valid == 2

