"""Majority/plurality voting for answer aggregation."""

from collections import Counter
from typing import List, Optional, Tuple

from .parsing import normalize_answer


def majority_vote(
    answers: List[Optional[str]],
    generation_order: Optional[List[int]] = None,
) -> Tuple[Optional[str], int, dict]:
    """Perform majority/plurality voting on a list of answers.
    
    Tie-breaking rule:
    If multiple answers have the same highest count, the answer whose
    first occurrence appears earliest in the generation order is selected.
    This ensures deterministic results.
    
    Args:
        answers: List of normalized answer strings (None for unparseable).
        generation_order: Optional list of indices indicating the order.
            If None, uses the list order (0, 1, 2, ...).
    
    Returns:
        Tuple of (winning_answer, vote_count, full_vote_distribution).
        Returns (None, 0, {}) if all answers are None/unparseable.
    """
    # Filter out None/unparseable answers
    valid_answers = [(i, a) for i, a in enumerate(answers) if a is not None]
    
    if not valid_answers:
        return None, 0, {}
    
    # Count votes
    counter = Counter(a for _, a in valid_answers)
    vote_dist = dict(counter.most_common())
    
    max_count = max(counter.values())
    
    # Find all answers with the max count (potential ties)
    candidates = [ans for ans, count in counter.items() if count == max_count]
    
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        # Tie-breaking: choose the answer whose first occurrence
        # appears earliest in the generation order
        first_occurrence = {}
        for idx, ans in valid_answers:
            if ans in candidates and ans not in first_occurrence:
                first_occurrence[ans] = idx
        
        winner = min(candidates, key=lambda a: first_occurrence[a])
    
    return winner, max_count, vote_dist


def evaluate_at_n(
    answers: List[Optional[str]],
    n: int,
) -> Tuple[Optional[str], int, int, dict]:
    """Evaluate majority vote using the first N answers.
    
    Args:
        answers: Full list of answers (already in deterministic order).
        n: Number of answers to use.
    
    Returns:
        Tuple of (predicted_answer, vote_count, valid_count, vote_distribution).
    """
    subset = answers[:n]
    valid_count = sum(1 for a in subset if a is not None)
    predicted, vote_count, vote_dist = majority_vote(subset)
    return predicted, vote_count, valid_count, vote_dist
