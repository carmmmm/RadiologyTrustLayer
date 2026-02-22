"""Compute 0-100 safety score and severity label from alignment results."""
from typing import Literal

Label = Literal["supported", "uncertain", "not_assessable", "needs_review"]

PENALTY = {
    "supported": 0,
    "uncertain": 8,
    "not_assessable": 12,
    "needs_review": 25,
}


def compute_score(claims: list[dict]) -> tuple[int, str, dict]:
    """
    Args:
        claims: list of dicts with at least {"label": Label}
    Returns:
        (overall_score 0-100, severity "low"|"medium"|"high", flag_counts dict)
    """
    if not claims:
        return 100, "low", {"supported": 0, "uncertain": 0, "not_assessable": 0, "needs_review": 0}

    counts: dict[str, int] = {k: 0 for k in PENALTY}
    total_penalty = 0

    for claim in claims:
        label = claim.get("label", "uncertain")
        if label not in PENALTY:
            label = "uncertain"
        counts[label] += 1
        total_penalty += PENALTY[label]

    max_possible = PENALTY["needs_review"] * len(claims)
    if max_possible == 0:
        raw_score = 100
    else:
        raw_score = max(0, 100 - int((total_penalty / max_possible) * 100))

    if raw_score >= 80:
        severity = "low"
    elif raw_score >= 50:
        severity = "medium"
    else:
        severity = "high"

    return raw_score, severity, counts


def severity_color(severity: str) -> str:
    return {"low": "green", "medium": "orange", "high": "red"}.get(severity, "gray")


def label_badge(label: str) -> str:
    dots = {
        "supported": '<span class="rtl-dot rtl-dot-green"></span>',
        "uncertain": '<span class="rtl-dot rtl-dot-amber"></span>',
        "not_assessable": '<span class="rtl-dot rtl-dot-blue"></span>',
        "needs_review": '<span class="rtl-dot rtl-dot-red"></span>',
    }
    return dots.get(label, '<span class="rtl-dot" style="background:#9aa0a6;"></span>')
