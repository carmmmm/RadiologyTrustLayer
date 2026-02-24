"""
Reusable HTML components for the Radiology Trust Layer UI.

Provides rendering functions for the score gauge, flag counts,
claim analysis table, and rewrite suggestion cards. All components
follow the RTL design system (Google-inspired Material style).
"""
from core.scoring.score import label_badge


def score_gauge_html(score: int, severity: str) -> str:
    """Render the circular safety score gauge with severity chip and progress bar."""
    color_map = {"low": "#34a853", "medium": "#fbbc04", "high": "#ea4335"}
    sev_bg = {"low": "#e6f4ea", "medium": "#fef7e0", "high": "#fce8e6"}
    sev_fg = {"low": "#137333", "medium": "#b06000", "high": "#c5221f"}
    color = color_map.get(severity, "#9aa0a6")
    pct = max(0, min(100, score))
    return f'''
    <div class="rtl-score-card">
      <div class="rtl-score-number" style="color:{color};">{score}</div>
      <div class="rtl-score-label">Safety Score / 100</div>
      <span class="rtl-severity-chip" style="background:{sev_bg.get(severity, '#f1f3f4')};color:{sev_fg.get(severity, '#5f6368')};">
        {severity} severity
      </span>
      <div class="rtl-score-bar-bg">
        <div class="rtl-score-bar-fill" style="background:{color};width:{pct}%;"></div>
      </div>
    </div>'''


def flag_counts_html(flag_counts: dict) -> str:
    """Render the label distribution summary (supported/uncertain/needs_review counts)."""
    items = [
        ("supported", "Supported", "#34a853", "rtl-dot-green"),
        ("uncertain", "Uncertain", "#fbbc04", "rtl-dot-amber"),
        ("needs_review", "Needs Review", "#ea4335", "rtl-dot-red"),
    ]
    parts = []
    for key, label, color, dot_class in items:
        n = flag_counts.get(key, 0)
        parts.append(f'''
        <div class="rtl-flag-item">
          <div class="rtl-flag-count" style="color:{color};">{n}</div>
          <div class="rtl-flag-label">
            <span class="rtl-dot {dot_class}"></span>{label}
          </div>
        </div>''')
    return f'<div class="rtl-flags">{"".join(parts)}</div>'


def claim_table_html(alignments: list[dict]) -> str:
    """Render the claim analysis table with label badges, confidence, and evidence."""
    if not alignments:
        return "<p style='color:#5f6368;'>No claims found.</p>"

    badge_class = {
        "supported": "rtl-badge-supported",
        "uncertain": "rtl-badge-uncertain",
        "needs_review": "rtl-badge-needs-review",
    }

    rows = []
    for a in alignments:
        badge = label_badge(a.get("label", "uncertain"))
        label = a.get("label", "uncertain").replace("_", " ").title()
        conf = a.get("confidence", 0)
        text = a.get("claim_text", a.get("claim_id", ""))
        evidence = a.get("evidence", "")
        conf_pct = f"{int(conf * 100)}%"
        cls = badge_class.get(a.get("label", ""), "")

        rows.append(f'''
        <tr>
          <td class="rtl-td">{text}</td>
          <td class="rtl-td"><span class="rtl-label-badge {cls}">{badge} {label}</span></td>
          <td class="rtl-td" style="color:#5f6368;">{conf_pct}</td>
          <td class="rtl-td" style="color:#3c4043;">{evidence}</td>
        </tr>''')

    return f'''
    <table class="rtl-table">
      <thead>
        <tr>
          <th class="rtl-th">Claim</th>
          <th class="rtl-th">Label</th>
          <th class="rtl-th">Confidence</th>
          <th class="rtl-th">Evidence</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>'''


def rewrite_suggestions_html(rewrites: list[dict]) -> str:
    """Render rewrite suggestion cards. Needs-review items show 'verify' actions; uncertain items show rewrites."""
    if not rewrites:
        return '<p style="color:#34a853;">No rewrites suggested — all claims are well-calibrated.</p>'

    parts = []
    for rw in rewrites:
        suggested = rw.get('suggested', '')
        is_verify = suggested.startswith("Verify with radiologist")
        if is_verify:
            label_text = "Action needed"
            label_color = "#c5221f"
            suggestion_style = 'color:#c5221f;font-weight:500;'
        else:
            label_text = "Suggested rewrite"
            label_color = "#137333"
            suggestion_style = ''
        parts.append(f'''
        <div class="rtl-rewrite-card">
          <div class="rtl-rewrite-label" style="color:#5f6368;">Original</div>
          <div style="color:#202124;margin-bottom:8px;">"{rw.get('original', '')}"</div>
          <div class="rtl-rewrite-label" style="color:{label_color};">{label_text}</div>
          <div class="rtl-rewrite-suggested" style="{suggestion_style}">"{suggested}"</div>
          <div class="rtl-rewrite-reason">Reason: {rw.get('reason', '')}</div>
        </div>''')

    return "".join(parts)
