# rl/instruction_split.py
from __future__ import annotations
from typing import List, Dict

def split_with_parser(paragraph: str, max_n: int, parse_instructions_fn=None) -> List[str]:
    """
    Uses your (or provided) parsers.py style splitter returning {goals, tips}.
    If parse_instructions_fn is None, fallback to naive rule split.
    """
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return ["(no instruction)"]

    if parse_instructions_fn is not None:
        parts: Dict[str, List[str]] = parse_instructions_fn(paragraph)
        goals = parts.get("goals", []) or []
        tips  = parts.get("tips", []) or []
        cands = goals + tips
        if not cands:
            cands = [paragraph]
        return cands[:max_n]

    # fallback: split by lines / periods
    seps = ["\n", ".", ";"]
    cands = [paragraph]
    for sep in seps:
        tmp = []
        for s in cands:
            tmp.extend([x.strip() for x in s.split(sep) if x.strip()])
        cands = tmp if tmp else cands
    return (cands[:max_n] if cands else [paragraph])

def split_with_lm(paragraph: str, max_n: int) -> List[str]:
    """
    Placeholder interface for LLM-based splitting.
    Keep it deterministic/stub by default (so code runs without external API).
    You can later plug an LLM call here.
    """
    # For now, just reuse rule split to keep runnable.
    return split_with_parser(paragraph, max_n, parse_instructions_fn=None)

