"""Answer extraction and normalization for model and gold answers."""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    r"""Normalize a numerical answer for comparison.
    
    Handles common formatting differences:
    - Commas in numbers: 1,200 -> 1200
    - Dollar signs: $42 or \$42 -> 42
    - Markdown bold/code: **42** or `42` -> 42
    - LaTeX wrappers: \\boxed{42} or \\text{42} -> 42
    - Angle brackets: <42> -> 42
    - Trailing dots: 42. -> 42
    - Trailing units: "42 dollars" -> "42", "42 clips" -> "42"
    - Trailing zeros: 42.0 -> 42, 42.00 -> 42
    - Percentage signs: 42% -> 42
    - Handles inf / nan as text without crashing.
    - Does not merge 1/2 with 0.5.
    
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
    
    # Strip markdown and bracket wrappers: **, `, <>, \$
    text = text.replace("**", "").replace("`", "")
    text = text.replace("\\$", "").replace("$", "")
    text = text.replace("%", "").replace(",", "")
    
    # Strip \boxed{...} and \text{...} wrappers (handling possible nesting)
    while True:
        new_text = re.sub(r'\\(boxed|text)\{([^}]*)\}', r'\2', text)
        if new_text == text:
            break
        text = new_text
    
    # Strip angle brackets < >
    text = text.replace("<", "").replace(">", "").strip()
    
    # Strip trailing dot
    text = text.rstrip(".").strip()
    
    # Strip trailing unit words ("42 clips" -> "42", "5 hours" -> "5", etc.)
    text = re.sub(r'(?<=\d)\s+[a-zA-Z/]+(?:\s+[a-zA-Z/]+)*\.?$', '', text)
    text = text.rstrip(".").strip()
    
    if not text:
        return None
    
    # Check for inf / nan text
    lower = text.lower()
    if lower in ("inf", "+inf", "infinity", "+infinity"):
        return "inf"
    if lower in ("-inf", "-infinity"):
        return "-inf"
    if lower == "nan":
        return "nan"
    
    # Try Decimal parsing
    try:
        d = Decimal(text)
        if d.is_infinite():
            return "-inf" if d < 0 else "inf"
        if d.is_nan():
            return "nan"
        if d == d.to_integral():
            return str(int(d))
        else:
            s = f"{d:f}".rstrip("0").rstrip(".")
            return s
    except (InvalidOperation, ValueError):
        # Don't crash on unparseable / fractions like "1/2"
        return text if text else None


def extract_model_answer(response: str) -> Tuple[Optional[str], str]:
    """Extract the final answer from a model response.
    
    Tries multiple extraction methods in order of preference:
    1. FINAL_ANSWER: <answer> (including **FINAL_ANSWER:** and FINAL_ANSWER**:)
    2. FINAL ANSWER: <answer> (without underscore)
    3. #### <answer> (GSM8K format)
    4. \\boxed{<answer>} (LaTeX format)
    
    Uses the LAST match per method and skips placeholder "<answer>".
    
    Args:
        response: The full model response text.
    
    Returns:
        Tuple of (extracted_answer, extraction_method).
        If extraction fails, returns (None, "unparseable").
    """
    if not response or not str(response).strip():
        return None, "unparseable"
    
    text = str(response).strip()
    
    # Method 1: FINAL_ANSWER
    p1 = r'\*{0,2}FINAL_ANSWER\*{0,2}:\*{0,2}\s*(.+?)(?:\n|$)'
    matches1 = list(re.finditer(p1, text, re.IGNORECASE))
    for m in reversed(matches1):
        raw = m.group(1).strip()
        cleaned_raw = raw.replace("*", "").strip().lower()
        if cleaned_raw in ("<answer>", "answer"):
            continue
        normalized = normalize_answer(raw)
        if normalized is not None and normalized.lower() != "answer":
            return normalized, "FINAL_ANSWER"
    
    # Method 2: FINAL_ANSWER_SPACE
    p2 = r'\*{0,2}FINAL\s+ANSWER\*{0,2}:\*{0,2}\s*(.+?)(?:\n|$)'
    matches2 = list(re.finditer(p2, text, re.IGNORECASE))
    for m in reversed(matches2):
        raw = m.group(1).strip()
        cleaned_raw = raw.replace("*", "").strip().lower()
        if cleaned_raw in ("<answer>", "answer"):
            continue
        normalized = normalize_answer(raw)
        if normalized is not None and normalized.lower() != "answer":
            return normalized, "FINAL_ANSWER_SPACE"
    
    # Method 3: HASH_MARKER (GSM8K style)
    p3 = r'####\s*(.+?)(?:\n|$)'
    matches3 = list(re.finditer(p3, text))
    for m in reversed(matches3):
        raw = m.group(1).strip()
        cleaned_raw = raw.replace("*", "").strip().lower()
        if cleaned_raw in ("<answer>", "answer"):
            continue
        normalized = normalize_answer(raw)
        if normalized is not None and normalized.lower() != "answer":
            return normalized, "HASH_MARKER"
    
    # Method 4: BOXED (LaTeX)
    p4 = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches4 = list(re.finditer(p4, text))
    for m in reversed(matches4):
        raw = m.group(1).strip()
        cleaned_raw = raw.replace("*", "").strip().lower()
        if cleaned_raw in ("<answer>", "answer"):
            continue
        normalized = normalize_answer(raw)
        if normalized is not None and normalized.lower() != "answer":
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
