"""Answer extraction and normalization for model and gold answers."""

import re
from typing import Optional, Tuple


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize a numerical answer for comparison.
    
    Handles common formatting differences:
    - Commas in numbers: 1,200 -> 1200
    - Dollar signs: $42 -> 42
    - Trailing units: "42 dollars" -> "42"
    - Trailing zeros: 42.0 -> 42, 42.00 -> 42
    - Whitespace
    - Negative signs
    - Percentage signs: 42% -> 42
    
    Args:
        answer: Raw answer string.
    
    Returns:
        Normalized answer string, or None if answer is empty/invalid.
    """
    if answer is None:
        return None
    
    text = str(answer).strip()
    
    if not text:
        return None
    
    # Remove dollar signs
    text = text.replace("$", "")
    
    # Remove percentage signs
    text = text.replace("%", "")
    
    # Remove commas from numbers
    text = text.replace(",", "")
    
    # Remove common trailing units/words
    text = re.sub(
        r'\s*(dollars?|cents?|pounds?|euros?|miles?|hours?|minutes?|'
        r'seconds?|meters?|feet|inches?|kg|lbs?|liters?|gallons?|'
        r'pieces?|items?|people|students?|books?|apples?|oranges?|'
        r'cookies?|candies|candy|marbles?|toys?|shirts?|pairs?|'
        r'pages?|days?|weeks?|months?|years?)\s*$',
        '', text, flags=re.IGNORECASE
    )
    
    text = text.strip()
    
    # Try to parse as a number and re-format
    try:
        num = float(text)
        # If it's an integer value, represent as int
        if num == int(num):
            return str(int(num))
        else:
            return str(num)
    except ValueError:
        # If we can't parse it as a number, return the cleaned text
        # This handles edge cases like fractions
        return text if text else None


def extract_model_answer(response: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from a model response.
    
    Tries multiple extraction methods in order of preference:
    1. FINAL_ANSWER: <answer>
    2. FINAL ANSWER: <answer> (without underscore)
    3. #### <answer> (GSM8K format)
    4. \\boxed{<answer>} (LaTeX format)
    
    Args:
        response: The full model response text.
    
    Returns:
        Tuple of (extracted_answer, extraction_method).
        If extraction fails, returns (None, "unparseable").
    """
    if not response or not response.strip():
        return None, "unparseable"
    
    text = response.strip()
    
    # Method 1: FINAL_ANSWER: <answer>
    match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        normalized = normalize_answer(raw)
        if normalized is not None:
            return normalized, "FINAL_ANSWER"
    
    # Method 2: FINAL ANSWER: <answer> (space instead of underscore)
    match = re.search(r'FINAL\s+ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        normalized = normalize_answer(raw)
        if normalized is not None:
            return normalized, "FINAL_ANSWER_SPACE"
    
    # Method 3: #### <answer> (GSM8K style)
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        raw = match.group(1).strip()
        normalized = normalize_answer(raw)
        if normalized is not None:
            return normalized, "HASH_MARKER"
    
    # Method 4: \boxed{<answer>} (LaTeX)
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        raw = match.group(1).strip()
        normalized = normalize_answer(raw)
        if normalized is not None:
            return normalized, "BOXED"
    
    return None, "unparseable"


def answers_match(answer1: Optional[str], answer2: Optional[str]) -> bool:
    """Check if two normalized answers match.
    
    Args:
        answer1: First normalized answer.
        answer2: Second normalized answer.
    
    Returns:
        True if answers are equivalent.
    """
    if answer1 is None or answer2 is None:
        return False
    
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    if norm1 is None or norm2 is None:
        return False
    
    return norm1 == norm2
