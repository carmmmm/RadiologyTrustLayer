"""
Render a radiology report with sentence-level color highlighting.

Each claim in the report is highlighted according to its alignment label:
  - Green: supported by imaging evidence
  - Yellow: uncertain (insufficient visual evidence)
  - Red: needs clinical review (possible mismatch)

Highlights are positioned using character-level spans from claim extraction.
"""
import html


LABEL_STYLE = {
    "supported": "background:#dcfce7;border-bottom:2px solid #16a34a;",
    "uncertain": "background:#fef9c3;border-bottom:2px solid #ca8a04;",
    "needs_review": "background:#fee2e2;border-bottom:2px solid #dc2626;",
}

LABEL_TOOLTIP = {
    "supported": "Supported by imaging evidence",
    "uncertain": "Uncertain — insufficient visual evidence",
    "needs_review": "Needs clinical review — possible mismatch",
}


def render_highlighted_report(report_text: str, alignments: list[dict], claims: list[dict]) -> str:
    """
    Return HTML of the report with each claim highlighted by its alignment label.
    """
    if not alignments or not claims:
        return f"<pre style='white-space:pre-wrap;'>{html.escape(report_text)}</pre>"

    # Build (start, end, label) spans sorted by start
    claim_map = {c["claim_id"]: c for c in claims}
    alignment_map = {a["claim_id"]: a["label"] for a in alignments}

    spans = []
    for claim in claims:
        cid = claim["claim_id"]
        label = alignment_map.get(cid, "uncertain")
        span = claim.get("sentence_span", {})
        start = span.get("start", 0)
        end = span.get("end", len(report_text))
        spans.append((start, end, label, cid))

    spans.sort(key=lambda x: x[0])

    # Build highlighted HTML
    result = []
    cursor = 0
    for start, end, label, claim_id in spans:
        # Text before this span
        if cursor < start:
            result.append(html.escape(report_text[cursor:start]))

        style = LABEL_STYLE.get(label, "")
        tooltip = LABEL_TOOLTIP.get(label, label)
        snippet = html.escape(report_text[start:end])
        result.append(
            f'<span style="padding:1px 2px;border-radius:3px;cursor:help;{style}" '
            f'title="{tooltip} (Claim {claim_id})">{snippet}</span>'
        )
        cursor = end

    # Remaining text
    if cursor < len(report_text):
        result.append(html.escape(report_text[cursor:]))

    body = "".join(result)
    legend = _legend_html()
    return f"""
    <div style="font-family:Georgia,serif;line-height:1.7;font-size:0.95rem;white-space:pre-wrap;padding:16px;background:white;border:1px solid #e5e7eb;border-radius:8px;">
    {body}
    </div>
    {legend}"""


def _legend_html() -> str:
    items = [
        ("Supported", "rtl-dot-green", LABEL_STYLE["supported"]),
        ("Uncertain", "rtl-dot-amber", LABEL_STYLE["uncertain"]),
        ("Needs Review", "rtl-dot-red", LABEL_STYLE["needs_review"]),
    ]
    parts = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;'
        f'border-radius:4px;font-size:0.75rem;color:#5f6368;{style}">'
        f'<span class="rtl-dot {dot_class}"></span>{label}</span>'
        for label, dot_class, style in items
    )
    return f'<div style="display:flex;gap:16px;justify-content:center;margin-top:10px;flex-wrap:wrap;">{parts}</div>'
