"""
Radiology Trust Layer (RTL) — Gradio Spaces Application.

This is the main entrypoint for the RTL web application, deployed on Hugging Face Spaces.
It provides a multi-page interface for auditing radiology reports against imaging evidence
using Google's MedGemma-4B-IT model with a custom LoRA adapter.

Pages:
  - Landing: Overview, pipeline diagram, and key metrics
  - Demo: Pre-loaded chest X-ray cases with pre-computed results
  - Single Audit: Upload an image + report for real-time pipeline execution
  - Batch Audit: Process multiple cases from a ZIP archive
  - History: Browse and filter past audit runs (requires login)
  - Evaluation: Before/after LoRA metrics and model card
  - Settings: Runtime configuration display

Navigation uses gr.Group visibility toggling with a persistent top navigation bar.
"""
import sys
import json
import logging
import tempfile
from pathlib import Path

# Ensure project root is on sys.path when run from spaces_app/ directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from PIL import Image

from core import config
from core.db.db import get_conn, init_db
from core.db.repo import (
    create_user, authenticate_user, get_user_display_name,
    list_recent_runs_for_user, list_all_runs_for_user,
    create_run, list_events_for_run,
    create_batch, update_batch_progress, link_batch_run,
)
from core.audit_trail.events import EventType, log as log_ev
from core.util.files import write_json, read_json
from scripts.init_space_storage import ensure_space_storage
from spaces_app.ui.components import (
    score_gauge_html, flag_counts_html, claim_table_html, rewrite_suggestions_html
)
from spaces_app.ui.render_report import render_highlighted_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "Radiology Trust Layer"
STORAGE_DIR = config.STORAGE_DIR
DB_PATH = config.DB_PATH
SCHEMA_PATH = _ROOT / "core" / "db" / "schema.sql"

PAGES = ["landing", "demo", "login", "home", "single", "batch", "detail", "history", "evaluation", "settings"]


# ─────────────────────────── CSS Design System ───────────────────────────────

RTL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Google-style design system for RTL */

/* Full-bleed — no white edges */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 24px !important;
    background: linear-gradient(180deg, #f0f2f5 0%, #e8eaed 50%, #f0f2f5 100%) !important;
    min-height: 100vh;
}

/* Tab styling */
.gradio-container .tab-nav button.selected {
    color: #1a73e8 !important;
    border-color: #1a73e8 !important;
}

/* Navigation Bar */
.rtl-nav-bar {
    display: flex !important;
    align-items: center !important;
    gap: 2px !important;
    padding: 6px 12px !important;
    background: #f8f9fa !important;
    border-bottom: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    margin-bottom: 16px !important;
}
.rtl-nav-bar button {
    background: none !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 18px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #5f6368 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    min-width: auto !important;
}
.rtl-nav-bar button:hover {
    background: #e8eaed !important;
    color: #202124 !important;
}

/* Typography */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #202124 !important;
    letter-spacing: -0.01em;
}
.gradio-container h2 {
    font-size: 1.375rem !important;
    font-weight: 500 !important;
}

/* Fix code blocks visibility (dark-on-dark bug) */
.gradio-container code {
    background: #f1f3f4 !important;
    color: #202124 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
}
.gradio-container pre code {
    padding: 12px !important;
    display: block !important;
}

/* Primary buttons */
button.primary {
    background: #1a73e8 !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
}
button.primary:hover {
    background: #1765cc !important;
}
button.secondary {
    border-radius: 20px !important;
}

/* Status Alerts */
.rtl-alert {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 8px 0;
}
.rtl-alert-success { background: #e6f4ea; color: #137333; border: 1px solid #ceead6; }
.rtl-alert-error { background: #fce8e6; color: #c5221f; border: 1px solid #f5c6cb; }
.rtl-alert-info { background: #e8f0fe; color: #1967d2; border: 1px solid #d2e3fc; }

/* Semantic Dots */
.rtl-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}
.rtl-dot-green { background: #34a853; }
.rtl-dot-amber { background: #fbbc04; }
.rtl-dot-blue  { background: #4285f4; }
.rtl-dot-red   { background: #ea4335; }

/* Score Card */
.rtl-score-card {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
}
.rtl-score-number {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}
.rtl-score-label {
    font-size: 0.8125rem;
    color: #5f6368;
    margin-top: 4px;
}
.rtl-severity-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 8px;
}
.rtl-score-bar-bg {
    background: #e0e0e0;
    border-radius: 4px;
    height: 6px;
    margin-top: 12px;
    overflow: hidden;
}
.rtl-score-bar-fill {
    height: 6px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Flag Counts */
.rtl-flags {
    display: flex;
    gap: 12px;
    justify-content: center;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    margin-top: 12px;
}
.rtl-flag-item { text-align: center; min-width: 70px; }
.rtl-flag-count { font-size: 1.5rem; font-weight: 600; line-height: 1; }
.rtl-flag-label {
    font-size: 0.6875rem;
    color: #5f6368;
    margin-top: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
}

/* Tables */
.rtl-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.rtl-table thead tr {
    background: #f8f9fa;
    border-bottom: 2px solid #e0e0e0;
}
.rtl-th {
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    color: #5f6368;
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.rtl-td {
    padding: 10px 12px;
    border-bottom: 1px solid #f1f3f4;
    color: #202124;
    font-size: 0.875rem;
}
.rtl-table tbody tr:hover { background: #f8f9fa; }

/* Label Badges */
.rtl-label-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8125rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 12px;
    white-space: nowrap;
}
.rtl-badge-supported { background: #e6f4ea; color: #137333; }
.rtl-badge-uncertain { background: #fef7e0; color: #b06000; }
.rtl-badge-needs-review { background: #fce8e6; color: #c5221f; }

/* Rewrite Cards */
.rtl-rewrite-card {
    margin-bottom: 12px;
    padding: 16px;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}
.rtl-rewrite-label {
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.rtl-rewrite-suggested {
    background: #e6f4ea;
    border: 1px solid #ceead6;
    padding: 8px 12px;
    border-radius: 6px;
    color: #137333;
    margin: 4px 0 8px 0;
}
.rtl-rewrite-reason {
    font-size: 0.8125rem;
    color: #5f6368;
    font-style: italic;
}

/* PHI Banner */
.rtl-phi-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: #e8f0fe;
    border-radius: 8px;
    font-size: 0.8125rem;
    color: #1967d2;
    margin-bottom: 12px;
}

/* Mode Chip */
.rtl-mode-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 12px;
    background: #f1f3f4;
    color: #5f6368;
}
.rtl-mode-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.rtl-mode-mock { background: #fbbc04; }
.rtl-mode-live { background: #34a853; }

/* Login Container */
.rtl-login-wrap { max-width: 420px; margin: 40px auto; }

/* Remove ALL group borders/backgrounds for seamless look */
.gradio-container .group,
.gradio-container .gr-group,
.gradio-container > .contain > div {
    border: none !important;
    padding: 0 !important;
    background: none !important;
    box-shadow: none !important;
}

/* Demo card buttons — styled as clickable cards */
.rtl-demo-card-btn {
    flex: 1 !important;
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 12px !important;
    padding: 20px 16px !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: box-shadow 0.2s, border-color 0.2s !important;
    min-height: 90px !important;
    font-size: 0.875rem !important;
    color: #202124 !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
}
.rtl-demo-card-btn:hover {
    box-shadow: 0 1px 6px rgba(32,33,36,0.15) !important;
    border-color: #1a73e8 !important;
}
.rtl-demo-card-btn span {
    white-space: pre-line !important;
    text-align: left !important;
}

/* Nav login button */
.rtl-nav-login {
    color: #1a73e8 !important;
    font-weight: 600 !important;
    margin-left: auto !important;
}

/* ── Landing Page ────────────────────────────────────────────────────── */

.rtl-landing {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 20px 0 20px;
    min-height: 80vh;
}

.rtl-landing-header {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding: 0 0 32px 0;
}

.rtl-landing-brand {
    font-size: 1.1rem;
    color: #5f6368;
    letter-spacing: 0.02em;
}
.rtl-landing-brand strong {
    color: #202124;
    font-weight: 600;
}

.rtl-landing-content {
    display: flex;
    gap: 72px;
    align-items: flex-start;
    padding: 0 2%;
}

.rtl-landing-diagram {
    flex: 0 0 320px;
}

.rtl-landing-text {
    flex: 1;
    min-width: 0;
}

.rtl-landing-text h1 {
    font-size: 2.5rem !important;
    font-weight: 400 !important;
    color: #202124 !important;
    margin: 0 0 20px 0 !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.rtl-landing-text p {
    color: #3c4043;
    line-height: 1.8;
    font-size: 0.95rem;
    margin-bottom: 16px;
}

/* Pipeline Architecture Diagram */
.rtl-model-callout {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: linear-gradient(135deg, #e8f0fe, #f8f9fa);
    border: 1.5px solid #d2e3fc;
    border-radius: 16px;
    margin-bottom: 0;
}
.rtl-model-callout-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #202124;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.65rem;
    flex-shrink: 0;
}
.rtl-model-callout-text {
    font-size: 0.72rem;
    color: #3c4043;
    line-height: 1.3;
}
.rtl-model-callout-text strong {
    color: #202124;
    display: block;
    font-size: 0.78rem;
}

.rtl-pipeline-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #5f6368;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}

.rtl-pipeline-steps {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.rtl-pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: #fff;
    border: 1.5px solid #dadce0;
    border-radius: 14px;
    font-size: 0.82rem;
    color: #202124;
    font-weight: 500;
}
.rtl-pipeline-step:hover {
    border-color: #1a73e8;
    box-shadow: 0 1px 4px rgba(26,115,232,0.12);
}

.rtl-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #202124;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
}

.rtl-pipeline-connector {
    width: 2px;
    height: 6px;
    background: #dadce0;
    margin-left: 25px;
}

/* Metrics Strip — removed, now inline */

/* Disclaimer Badge */
.rtl-landing-disclaimer {
    margin: 20px 0 0 0;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #5f6368;
}
.rtl-disclaimer-badge {
    background-color: #5f6368;
    color: white;
    padding: 3px 14px;
    border-radius: 16px;
    font-size: 0.8rem;
    font-weight: 500;
    white-space: nowrap;
    margin-right: 6px;
    display: inline-block;
    transform: translateY(-1px);
}

/* CTA Button Row */
.rtl-landing-cta-row {
    justify-content: flex-start !important;
    gap: 12px !important;
    padding: 24px 0 16px 0 !important;
}
.rtl-cta-btn {
    background-color: #202124 !important;
    color: white !important;
    padding: 12px 32px !important;
    border: none !important;
    border-radius: 25px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    min-width: 160px !important;
    cursor: pointer !important;
}
.rtl-cta-btn:hover {
    background-color: #3c4043 !important;
}
.rtl-cta-secondary {
    background: none !important;
    border: 1.5px solid #dadce0 !important;
    border-radius: 25px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #202124 !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
.rtl-cta-secondary:hover {
    background: #f8f9fa !important;
}

/* Nav Bar — seamless, no background */
.rtl-nav-bar-v2 {
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    padding: 16px 24px 8px 24px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    margin-bottom: 0 !important;
}
.rtl-nav-bar-v2 button {
    background: transparent !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 18px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #3c4043 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    min-width: auto !important;
    transition: all 0.15s ease !important;
}
.rtl-nav-bar-v2 button:hover {
    background: rgba(0,0,0,0.05) !important;
    color: #202124 !important;
}
.rtl-nav-logo {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #202124 !important;
    letter-spacing: 0.02em !important;
    margin-right: 12px !important;
}

/* Nav page info bar — hidden */
.rtl-nav-page-info {
    display: none !important;
}

/* Pipeline accordions — hidden, not used */

/* Consistent page height — prevents layout shifts */
.gradio-container .group {
    min-height: 0 !important;
}

/* Dummy history overlay */
.rtl-history-placeholder {
    position: relative;
    margin-top: 16px;
}
.rtl-history-placeholder table {
    opacity: 0.35;
    pointer-events: none;
}
.rtl-history-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    z-index: 5;
}
.rtl-history-overlay h3 {
    font-size: 1rem !important;
    margin: 0 0 8px 0 !important;
}
.rtl-history-overlay p {
    color: #5f6368;
    font-size: 0.85rem;
    margin: 0;
}

/* Hide Gradio's built-in progress bars and error toasts — we use our own UI */
.gradio-container .progress-bar,
.gradio-container .progress-text,
.gradio-container .meta-text,
.gradio-container .progress-level {
    display: none !important;
}
.toast-wrap, .toast-body, .toast-close {
    display: none !important;
}

/* Loading spinner */
@keyframes rtl-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes rtl-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.rtl-loading {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    background: #e8f0fe;
    border-radius: 20px;
    color: #1967d2;
    font-size: 0.82rem;
    font-weight: 500;
    animation: rtl-pulse 2s ease-in-out infinite;
}
.rtl-loading-spinner {
    width: 16px;
    height: 16px;
    border: 2.5px solid #d2e3fc;
    border-top: 2.5px solid #1a73e8;
    border-radius: 50%;
    animation: rtl-spin 0.8s linear infinite;
    flex-shrink: 0;
}

/* Mobile */
@media (max-width: 768px) {
    .rtl-landing-content {
        flex-direction: column;
        gap: 32px;
    }
    .rtl-landing-diagram {
        flex: none;
        width: 100%;
    }
    .rtl-landing-text h1 {
        font-size: 1.8rem !important;
    }
    .rtl-metrics-strip {
        flex-wrap: wrap;
        gap: 16px;
    }
}
"""


# ─────────────────────────── Example cases ───────────────────────────────────

def _load_example_manifest() -> list[dict]:
    manifest = config.EXAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text())
    return data.get("examples", [])


EXAMPLE_CASES = _load_example_manifest()



def _load_example_case(index: int):
    if index >= len(EXAMPLE_CASES):
        return None, "", ""
    ex = EXAMPLE_CASES[index]
    img_path = _ROOT / ex["image_path"]
    rpt_path = _ROOT / ex["report_path"]
    image = Image.open(img_path) if img_path.exists() else None
    report = rpt_path.read_text().strip() if rpt_path.exists() else ""
    label = ex["label"]
    return image, label, report


def _load_preloaded_demo(index: int):
    """Load example case AND pre-computed mock results for instant demo display."""
    from core.pipeline.medgemma_client import _MOCK_PNEUMONIA, _MOCK_CHF, _MOCK_NORMAL
    from core.scoring.score import compute_score

    if index >= len(EXAMPLE_CASES):
        return (None, "", "") + ("",) * 8

    ex = EXAMPLE_CASES[index]
    img_path = _ROOT / ex["image_path"]
    rpt_path = _ROOT / ex["report_path"]
    image = Image.open(img_path) if img_path.exists() else None
    report = rpt_path.read_text().strip() if rpt_path.exists() else ""
    label = ex["label"]

    # Pick the right mock data based on case index
    mock_sets = [_MOCK_PNEUMONIA, _MOCK_CHF, _MOCK_NORMAL]
    mock = mock_sets[index] if index < len(mock_sets) else _MOCK_PNEUMONIA

    # Deep copy to avoid mutating module-level mock data
    import copy
    claims = copy.deepcopy(mock["claim_extraction"]["claims"])
    alignments = copy.deepcopy(mock["alignment"]["alignments"])

    # Recalculate sentence_span to match the actual report file text
    for claim in claims:
        text = claim.get("text", "")
        idx = report.find(text)
        if idx >= 0:
            claim["sentence_span"] = {"start": idx, "end": idx + len(text)}

    # Merge claim text into alignments
    claim_map = {c["claim_id"]: c for c in claims}
    for a in alignments:
        a["claim_text"] = claim_map.get(a.get("claim_id", ""), {}).get("text", "")

    overall_score, severity, flag_counts = compute_score(alignments)

    score_html = score_gauge_html(overall_score, severity)
    flag_html = flag_counts_html(flag_counts)
    report_html = render_highlighted_report(report, alignments, claims)
    claims_html = claim_table_html(alignments)
    rewrites_html = rewrite_suggestions_html(mock["rewrite"].get("rewrites", []))

    cs = mock.get("clinician_summary", {})
    clinician_md = (
        f"**Summary:** {cs.get('summary', '')}\n\n"
        f"**Recommendation:** {cs.get('recommendation', '').replace('_', ' ').title()}\n\n"
        + ("**Key Concerns:**\n" + "\n".join(f"- {c}" for c in cs.get("key_concerns", [])) if cs.get("key_concerns") else "")
        + f"\n\n*{cs.get('confidence_note', '')}*"
    )

    pe = mock.get("patient_explain", {})
    patient_md = (
        f"**Summary:**\n\n{pe.get('plain_language_summary', '')}\n\n"
        f"**What was found:** {pe.get('what_was_found', '')}\n\n"
        f"**What it means:** {pe.get('what_it_means', '')}\n\n"
        f"**Next steps:** {pe.get('next_steps', '')}"
    )

    return (image, label, report,
            score_html, flag_html, report_html, claims_html,
            rewrites_html, clinician_md, patient_md)


# ─────────────────────────── State helpers ────────────────────────────────

def _default_state() -> dict:
    return {
        "user_id": None,
        "run_id": None,
        "batch_id": None,
        "current_result": None,
        "page": "landing",
    }


PAGE_DESCRIPTIONS = {
    "landing": ("RTL", "Overview of the Radiology Trust Layer and its 6-step audit pipeline"),
    "demo": ("Demo", "Try pre-loaded chest X-ray cases to see RTL in action"),
    "single": ("Single Audit", "Upload a radiology image and report to run the full audit pipeline"),
    "batch": ("Batch", "Upload a ZIP archive of cases for bulk processing"),
    "history": ("History", "Browse and filter past audits (requires login)"),
    "evaluation": ("Evaluation", "Model card, LoRA before/after metrics, and example outputs"),
    "settings": ("Settings", "Current configuration and environment variables"),
    "detail": ("Audit Detail", "Full results for a single audit run"),
    "home": ("Home", "Recent audits and account overview"),
    "login": ("", ""),
}


def _page_info_html(page: str) -> str:
    name, desc = PAGE_DESCRIPTIONS.get(page, ("", ""))
    if not name:
        return ""
    return f'<div class="rtl-nav-page-info"><span class="rtl-page-name">{name}</span> {desc}</div>'


# Map page names to which nav button index (0-based) should be highlighted
# Order: RTL, Demo, Single, Batch, History, Evaluation, Settings
_NAV_ACTIVE = {
    "landing": 0,
    "demo": 1,
    "single": 2,
    "batch": 3,
    "history": 4,
    "evaluation": 5,
    "settings": 6,
    "home": 0,
    "detail": 2,
}
_NAV_COUNT = 7  # RTL, Demo, Single Audit, Batch, History, Evaluation, Settings


def _nav_btn_updates(page: str) -> tuple:
    """Return gr.update for each of the 7 nav buttons — active one gets variant='primary'."""
    active_idx = _NAV_ACTIVE.get(page, -1)
    return tuple(
        gr.update(variant="primary") if i == active_idx else gr.update(variant="secondary")
        for i in range(_NAV_COUNT)
    )


def _set_views(page: str) -> tuple:
    """Return gr.update() for nav_group + nav_page_info + nav_buttons + each view in PAGES order."""
    nav_visible = page != "login"
    return (
        gr.update(visible=nav_visible),
        gr.update(value=_page_info_html(page)),
    ) + _nav_btn_updates(page) + tuple(
        gr.update(visible=(page == p)) for p in PAGES
    )


def _after_login_data(state: dict) -> tuple[str, list]:
    """Return (header_html, recent_rows) for the home page."""
    conn = get_conn(DB_PATH)
    user_id = state["user_id"]
    name = get_user_display_name(conn, user_id) or "User"
    recent = list_recent_runs_for_user(conn, user_id, limit=8)

    mode_class = "rtl-mode-mock" if config.MEDGEMMA_MOCK else "rtl-mode-live"
    mode_text = "Mock" if config.MEDGEMMA_MOCK else "Live MedGemma"
    header_html = f'''
    <div style="margin-bottom:16px;">
        <h1 style="font-size:1.5rem;font-weight:500;color:#202124;margin:0 0 8px 0;">{APP_TITLE}</h1>
        <div style="display:flex;align-items:center;gap:16px;color:#5f6368;font-size:0.875rem;">
            <span>Signed in as <strong style="color:#202124;">{name}</strong></span>
            <span class="rtl-mode-chip">
                <span class="rtl-mode-dot {mode_class}"></span>{mode_text}
            </span>
            <span>Model: <code style="background:#f1f3f4;padding:2px 6px;border-radius:4px;font-size:0.8rem;">{config.MEDGEMMA_MODEL_ID}</code></span>
        </div>
    </div>'''
    return header_html, recent


# ─────────────────────────── Alert helpers ────────────────────────────────

def _alert(msg: str, kind: str = "info") -> str:
    return f'<div class="rtl-alert rtl-alert-{kind}">{msg}</div>'

def _loading_html(msg: str = "Running audit pipeline...") -> str:
    return f'<div class="rtl-loading"><div class="rtl-loading-spinner"></div>{msg}</div>'


def _loading_results_html(step: int = 1, total: int = 6) -> str:
    """Full-area loading indicator shown in the results column."""
    steps = [
        "Extracting claims from report",
        "Analyzing image for visual findings",
        "Aligning claims to image evidence",
        "Computing safety score",
        "Generating rewrite suggestions",
        "Building clinician summary",
    ]
    current = steps[min(step - 1, len(steps) - 1)] if step > 0 else "Initializing"
    pct = int((step / total) * 100)
    step_items = ""
    for i, s in enumerate(steps):
        if i + 1 < step:
            icon = '<span style="color:#137333;font-weight:700;">&#10003;</span>'
            color = "#3c4043"
            weight = "400"
        elif i + 1 == step:
            icon = '<span class="rtl-loading-spinner" style="width:14px;height:14px;border-width:2.5px;display:inline-block;vertical-align:middle;"></span>'
            color = "#1a73e8"
            weight = "600"
        else:
            icon = '<span style="color:#bdc1c6;">&#9679;</span>'
            color = "#80868b"
            weight = "400"
        step_items += f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;font-size:0.85rem;color:{color};font-weight:{weight};">{icon} {s}</div>'
    return f'''<div style="padding:40px 24px;text-align:center;">
  <div class="rtl-loading-spinner" style="width:40px;height:40px;border-width:3.5px;margin:0 auto 20px;"></div>
  <div style="font-size:1.2rem;font-weight:700;color:#202124;margin-bottom:6px;">Running Audit Pipeline</div>
  <div style="font-size:0.9rem;font-weight:500;color:#3c4043;margin-bottom:20px;">Step {step}/{total}: {current}...</div>
  <div style="background:#dadce0;border-radius:4px;height:8px;max-width:320px;margin:0 auto 24px;">
    <div style="background:#1a73e8;border-radius:4px;height:8px;width:{pct}%;transition:width 0.3s;"></div>
  </div>
  <div style="text-align:left;max-width:300px;margin:0 auto;">{step_items}</div>
</div>'''


# ─────────────────────────── Theme ──────────────────────────────────────

light_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
)
light_theme.set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    body_text_color="#202124",
    body_text_color_dark="#202124",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8f9fa",
    background_fill_secondary_dark="#f8f9fa",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e0e0e0",
    block_border_color_dark="#e0e0e0",
    block_label_text_color="#202124",
    block_label_text_color_dark="#202124",
    block_title_text_color="#202124",
    block_title_text_color_dark="#202124",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#dadce0",
    input_border_color_dark="#dadce0",
    input_placeholder_color="#9aa0a6",
    input_placeholder_color_dark="#9aa0a6",
    table_even_background_fill="#f8f9fa",
    table_even_background_fill_dark="#f8f9fa",
    table_odd_background_fill="#ffffff",
    table_odd_background_fill_dark="#ffffff",
    border_color_primary="#e0e0e0",
    border_color_primary_dark="#e0e0e0",
    panel_background_fill="#f8f9fa",
    panel_background_fill_dark="#f8f9fa",
    button_primary_background_fill="#1a73e8",
    button_primary_background_fill_dark="#1a73e8",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_text_color="#202124",
    button_secondary_text_color_dark="#202124",
    shadow_drop="none",
    shadow_drop_lg="none",
)

# ─────────────────────────── Main Blocks app ─────────────────────────────

def main() -> gr.Blocks:
    ensure_space_storage(storage_dir=STORAGE_DIR, db_path=DB_PATH)
    init_db(DB_PATH, SCHEMA_PATH)

    with gr.Blocks(title=APP_TITLE, theme=light_theme, css=RTL_CSS) as demo:
        state = gr.State(_default_state())

        # ═══════════════════════════════════════════════════════════════════
        # PERSISTENT NAVIGATION BAR (hidden on login page)
        # ═══════════════════════════════════════════════════════════════════
        nav_group = gr.Group(visible=False)
        with nav_group:
            with gr.Row(elem_classes=["rtl-nav-bar-v2"]):
                nav_home_btn = gr.Button("RTL", size="sm", elem_classes=["rtl-nav-logo"])
                nav_demo = gr.Button("Demo", size="sm")
                nav_single = gr.Button("Single Audit", size="sm")
                nav_batch = gr.Button("Batch", size="sm")
                nav_history = gr.Button("History", size="sm")
                nav_eval = gr.Button("Evaluation", size="sm")
                nav_settings = gr.Button("Settings", size="sm")
                gr.HTML('<span style="flex:1;"></span>')
                nav_login = gr.Button("Log in", size="sm", elem_classes=["rtl-nav-login"])
            nav_page_info = gr.HTML()

        # ═══════════════════════════════════════════════════════════════════
        # LANDING PAGE
        # ═══════════════════════════════════════════════════════════════════
        landing_view = gr.Group(visible=True)
        with landing_view:
            # Top bar: MG callout left | RTL title + MedGemma brand right (aligned to columns)
            gr.HTML('''<div style="display:flex;align-items:center;padding:24px 28px 12px 28px;gap:16px;">
  <div style="flex:1;min-width:280px;">
    <div class="rtl-model-callout" style="margin:0;">
      <div class="rtl-model-callout-icon">MG</div>
      <div class="rtl-model-callout-text"><strong>MedGemma 4B + RTL LoRA</strong>Open-weight medical AI with custom fine-tuning</div>
    </div>
  </div>
  <div style="flex:2;display:flex;align-items:center;justify-content:space-between;">
    <span style="font-size:1.8rem;font-weight:700;color:#202124;letter-spacing:-0.01em;">Radiology Trust Layer</span>
    <span style="font-size:1.15rem;color:#5f6368;letter-spacing:0.02em;">Med<strong style="color:#202124;font-weight:600;">Gemma</strong></span>
  </div>
</div>''')
            # Two-column layout: pipeline left, title + description right
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('''<div class="rtl-pipeline-label">6-Step Audit Pipeline</div>
<div class="rtl-pipeline-steps">
  <div class="rtl-pipeline-step"><span class="rtl-step-num">1</span> Claim Extraction</div>
  <div class="rtl-pipeline-connector"></div>
  <div class="rtl-pipeline-step"><span class="rtl-step-num">2</span> Image Findings</div>
  <div class="rtl-pipeline-connector"></div>
  <div class="rtl-pipeline-step"><span class="rtl-step-num">3</span> Alignment</div>
  <div class="rtl-pipeline-connector"></div>
  <div class="rtl-pipeline-step"><span class="rtl-step-num">4</span> Scoring</div>
  <div class="rtl-pipeline-connector"></div>
  <div class="rtl-pipeline-step"><span class="rtl-step-num">5</span> Rewrite Suggestions</div>
  <div class="rtl-pipeline-connector"></div>
  <div class="rtl-pipeline-step"><span class="rtl-step-num">6</span> Clinician Summary</div>
</div>''')
                with gr.Column(scale=2):
                    gr.HTML('''<div style="font-size:0.88rem;color:#3c4043;line-height:1.7;padding-top:32px;padding-right:60px;">
  <p style="margin-top:0;">Radiology Trust Layer is designed to audit radiology report language, not to generate diagnoses or replace clinical judgment.</p>
  <p>The system extracts every claim from a free-text radiology report, analyzes the corresponding medical image with MedGemma's vision encoder, and aligns each claim to visual findings. Claims are labeled as <em>supported</em>, <em>uncertain</em>, or <em>needs review</em>. Flagged claims receive suggested rewrites using calibrated uncertainty language — turning overconfident statements into properly hedged ones. Clinicians get a structured summary highlighting key concerns; patients get an accessible plain-language explanation of the findings.</p>
  <p style="margin-bottom:0;">Built on Google's MedGemma-4B-IT with a custom LoRA adapter fine-tuned for JSON schema compliance and uncertainty calibration.</p>
</div>''')
            # Large metrics strip with real training results
            gr.HTML('''<div style="display:flex;gap:0;padding:28px 0 12px 0;border-top:1px solid rgba(0,0,0,0.08);margin-top:12px;">
  <div style="flex:1;text-align:center;border-right:1px solid rgba(0,0,0,0.06);"><span style="font-size:2.4rem;font-weight:700;color:#202124;">100%</span><div style="font-size:0.78rem;color:#5f6368;margin-top:4px;">Schema Compliance</div></div>
  <div style="flex:1;text-align:center;border-right:1px solid rgba(0,0,0,0.06);"><span style="font-size:2.4rem;font-weight:700;color:#202124;">87.3%</span><div style="font-size:0.78rem;color:#5f6368;margin-top:4px;">Label Accuracy</div></div>
  <div style="flex:1;text-align:center;border-right:1px solid rgba(0,0,0,0.06);"><span style="font-size:2.4rem;font-weight:700;color:#202124;">6</span><div style="font-size:0.78rem;color:#5f6368;margin-top:4px;">Pipeline Steps</div></div>
  <div style="flex:1;text-align:center;"><span style="font-size:2.4rem;font-weight:700;color:#202124;">4B</span><div style="font-size:0.78rem;color:#5f6368;margin-top:4px;">Model Parameters</div></div>
</div><div style="font-size:0.72rem;color:#9aa0a6;padding:0 0 4px 0;text-align:center;">Evaluated on 50 synthetic radiology cases -- See Evaluation tab for full before/after breakdown</div>''')
            # Disclaimer
            gr.HTML('''<div class="rtl-landing-disclaimer"><span class="rtl-disclaimer-badge">Disclaimer</span> Research demonstration for the MedGemma Impact Challenge. Not for clinical use. Do not upload real patient data.</div>''')

        # ═══════════════════════════════════════════════════════════════════
        # DEMO VIEW — standalone page with 3 example cases
        # ═══════════════════════════════════════════════════════════════════
        demo_view = gr.Group(visible=False)
        with demo_view:
            gr.Markdown("## Demo")
            gr.Markdown("Select a pre-loaded chest X-ray case, then click **Run Audit** to see results. Results are pre-computed with MedGemma.")
            with gr.Row():
                demo_btn_1 = gr.Button(
                    "CXR 01 — Right Lower Lobe Pneumonia\nExpected severity: Low",
                    elem_classes=["rtl-demo-card-btn"],
                )
                demo_btn_2 = gr.Button(
                    "CXR 02 — Congestive Heart Failure\nExpected severity: Medium",
                    elem_classes=["rtl-demo-card-btn"],
                )
                demo_btn_3 = gr.Button(
                    "CXR 03 — Normal Study\nExpected severity: Low",
                    elem_classes=["rtl-demo-card-btn"],
                )
            with gr.Row():
                with gr.Column(scale=1):
                    demo_image = gr.Image(label="Radiology Image", type="pil", height=300, interactive=False)
                    demo_case_label = gr.Textbox(label="Case", interactive=False)
                    demo_report = gr.Textbox(label="Report", lines=6, interactive=False)
                    demo_run_btn = gr.Button("Run Audit", variant="primary", size="lg")
                with gr.Column(scale=1):
                    demo_score_html = gr.HTML('<p style="color:#5f6368;">Select a case above, then click Run Audit.</p>')
                    demo_flag_html = gr.HTML()
                    with gr.Tabs():
                        with gr.Tab("Report Highlights"):
                            demo_report_html = gr.HTML()
                        with gr.Tab("Claim Analysis"):
                            demo_claims_html = gr.HTML()
                        with gr.Tab("Suggested Rewrites"):
                            demo_rewrites_html = gr.HTML()
                        with gr.Tab("Clinician Summary"):
                            demo_clinician_md = gr.Markdown()
                        with gr.Tab("Patient Explanation"):
                            demo_patient_md = gr.Markdown()

        # ═══════════════════════════════════════════════════════════════════
        # LOGIN VIEW
        # ═══════════════════════════════════════════════════════════════════
        login_view = gr.Group(visible=False)
        with login_view:
            gr.HTML(f'''
            <div style="text-align:center;padding:32px 0 8px 0;">
                <h1 style="font-size:1.75rem;font-weight:500;color:#202124;margin-bottom:8px;">{APP_TITLE}</h1>
                <p style="font-size:0.875rem;color:#5f6368;margin-bottom:0;">MedGemma-powered radiology report auditing</p>
            </div>''')
            gr.HTML('''<div class="rtl-phi-banner">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="#1967d2" style="flex-shrink:0;">
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1s3.1 1.39 3.1 3.1v2z"/>
                </svg>
                Do not upload real patient data
            </div>''')
            with gr.Tab("Log In"):
                login_email = gr.Textbox(label="Email", placeholder="name@example.com")
                login_pw = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Log In", variant="primary")
                login_msg = gr.HTML()
            with gr.Tab("Create Account"):
                create_email = gr.Textbox(label="Email")
                create_name_box = gr.Textbox(label="Display name", placeholder="Dr. Smith")
                create_pw = gr.Textbox(label="Password", type="password")
                create_btn = gr.Button("Create Account", variant="primary")
                create_msg = gr.HTML()
            login_back = gr.Button("Back", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # HOME VIEW
        # ═══════════════════════════════════════════════════════════════════
        home_view = gr.Group(visible=False)
        with home_view:
            home_header = gr.HTML()
            gr.HTML('''<div class="rtl-phi-banner">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="#1967d2" style="flex-shrink:0;">
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1s3.1 1.39 3.1 3.1v2z"/>
                </svg>
                Public demo — do not upload patient data
            </div>''')
            gr.Markdown("### Recent Audits")
            recent_table = gr.Dataframe(
                headers=["Date", "Case Label", "Score", "Severity", "Run ID"],
                datatype=["str", "str", "number", "str", "str"],
                interactive=False, wrap=True,
            )
            with gr.Row():
                home_run_id_box = gr.Textbox(label="Open run by ID", placeholder="Paste run_id")
                home_open_btn = gr.Button("Open Audit Detail", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # SINGLE AUDIT VIEW
        # ═══════════════════════════════════════════════════════════════════
        single_view = gr.Group(visible=False)
        with single_view:
            gr.Markdown("## Single Study Audit")
            gr.Markdown(
                "Upload a radiology image and paste the corresponding report. "
                "RTL audits each claim against imaging evidence using MedGemma."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    single_image = gr.Image(label="Radiology Image", type="pil", height=300)
                    single_case_label = gr.Textbox(label="Case label (optional)")
                    single_report = gr.Textbox(
                        label="Radiology Report Text", lines=8,
                        placeholder="Paste the free-text radiology report here..."
                    )
                    single_lora = gr.Checkbox(label="Use RTL LoRA adapter", value=False)
                    single_run_btn = gr.Button("Run Audit", variant="primary", size="lg")
                    single_status = gr.HTML()

                with gr.Column(scale=1):
                    single_score_html = gr.HTML('<p style="color:#5f6368;">Results appear after audit.</p>')
                    single_flag_html = gr.HTML()
                    with gr.Tabs():
                        with gr.Tab("Report Highlights"):
                            single_report_html = gr.HTML()
                        with gr.Tab("Claim Analysis"):
                            single_claims_html = gr.HTML()
                        with gr.Tab("Suggested Rewrites"):
                            single_rewrites_html = gr.HTML()
                            single_accept_all = gr.Button("Accept All Rewrites", size="sm")
                        with gr.Tab("Clinician Summary"):
                            single_clinician_md = gr.Markdown()
                        with gr.Tab("Patient Explanation"):
                            single_patient_md = gr.Markdown()
                        with gr.Tab("Edited Report"):
                            single_edited_report = gr.Textbox(label="Edited Report", lines=8, interactive=False)

            single_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # BATCH AUDIT VIEW
        # ═══════════════════════════════════════════════════════════════════
        batch_view = gr.Group(visible=False)
        with batch_view:
            gr.Markdown("## Batch Audit")
            gr.Markdown("Upload a ZIP archive of radiology cases. Each case: one image + one .txt report.")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_zip = gr.File(label="Upload ZIP", file_types=[".zip"])
                    batch_label_box = gr.Textbox(label="Batch label (optional)")
                    batch_lora = gr.Checkbox(label="Use RTL LoRA adapter", value=False)
                    batch_run_btn = gr.Button("Run Batch Audit", variant="primary", size="lg")
                    batch_status = gr.HTML()
                with gr.Column(scale=2):
                    batch_progress_md = gr.Markdown()
                    batch_summary_md = gr.Markdown()
                    batch_table = gr.Dataframe(
                        headers=["Case ID", "Score", "Severity", "Supported", "Uncertain", "Needs Review", "Run ID"],
                        datatype=["str", "number", "str", "number", "number", "number", "str"],
                        interactive=False, wrap=True,
                    )
            batch_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # AUDIT DETAIL VIEW
        # ═══════════════════════════════════════════════════════════════════
        detail_view = gr.Group(visible=False)
        with detail_view:
            gr.Markdown("## Audit Detail")
            detail_run_md = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    detail_score_html = gr.HTML()
                    detail_flag_html = gr.HTML()
                    detail_meta_md = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Report Highlights"):
                            detail_report_html = gr.HTML()
                        with gr.Tab("Claim Analysis"):
                            detail_claims_html = gr.HTML()
                        with gr.Tab("Rewrites"):
                            detail_rewrites_html = gr.HTML()
                        with gr.Tab("Clinician Summary"):
                            detail_clinician_md = gr.Markdown()
                        with gr.Tab("Patient Explanation"):
                            detail_patient_md = gr.Markdown()
                        with gr.Tab("Edited Report"):
                            detail_edited_txt = gr.Textbox(label="Edited Report", lines=8, interactive=False)
                        with gr.Tab("Audit Trail"):
                            detail_trail_html = gr.HTML()
            with gr.Row():
                detail_export_btn = gr.Button("Export JSON", size="sm")
                detail_export_file = gr.File(label="Download", visible=False)
            detail_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # HISTORY VIEW
        # ═══════════════════════════════════════════════════════════════════
        history_view = gr.Group(visible=False)
        with history_view:
            gr.Markdown("## Audit History")
            hist_placeholder = gr.HTML('''
            <div class="rtl-history-placeholder">
              <table class="rtl-table" style="width:100%;">
                <thead><tr>
                  <th class="rtl-th">Date</th><th class="rtl-th">Case Label</th>
                  <th class="rtl-th">Score</th><th class="rtl-th">Severity</th>
                  <th class="rtl-th">Model</th><th class="rtl-th">Run ID</th>
                </tr></thead>
                <tbody>
                  <tr><td class="rtl-td">2026-02-22 14:32</td><td class="rtl-td">CXR - Right lower lobe</td><td class="rtl-td">82</td><td class="rtl-td">Low</td><td class="rtl-td">medgemma-4b-it</td><td class="rtl-td">a1b2c3d4</td></tr>
                  <tr><td class="rtl-td">2026-02-22 13:15</td><td class="rtl-td">CXR - CHF evaluation</td><td class="rtl-td">61</td><td class="rtl-td">Medium</td><td class="rtl-td">medgemma-4b-it</td><td class="rtl-td">e5f6g7h8</td></tr>
                  <tr><td class="rtl-td">2026-02-21 09:48</td><td class="rtl-td">CXR - Normal study</td><td class="rtl-td">94</td><td class="rtl-td">Low</td><td class="rtl-td">medgemma-4b-it</td><td class="rtl-td">i9j0k1l2</td></tr>
                  <tr><td class="rtl-td">2026-02-20 16:22</td><td class="rtl-td">CXR - Pleural effusion</td><td class="rtl-td">45</td><td class="rtl-td">High</td><td class="rtl-td">medgemma-4b-it</td><td class="rtl-td">m3n4o5p6</td></tr>
                  <tr><td class="rtl-td">2026-02-20 11:05</td><td class="rtl-td">CXR - Pneumothorax</td><td class="rtl-td">73</td><td class="rtl-td">Medium</td><td class="rtl-td">medgemma-4b-it</td><td class="rtl-td">q7r8s9t0</td></tr>
                </tbody>
              </table>
              <div class="rtl-history-overlay">
                <h3>You are not logged in</h3>
                <p>Log in or create an account to save your<br>audit history and access it across sessions.</p>
              </div>
            </div>
            ''')
            with gr.Row():
                hist_filter_sev = gr.Dropdown(choices=["All", "low", "medium", "high"], value="All", label="Severity")
                hist_filter_score = gr.Slider(0, 100, 0, step=5, label="Min score")
                hist_refresh = gr.Button("Refresh", size="sm")
            hist_table = gr.Dataframe(
                headers=["Date", "Case Label", "Score", "Severity", "Model", "Run ID"],
                datatype=["str", "str", "number", "str", "str", "str"],
                interactive=False, wrap=True, visible=False,
            )
            with gr.Row():
                hist_run_id_box = gr.Textbox(label="Run ID to open")
                hist_open_btn = gr.Button("Open Audit Detail", size="sm")
            history_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # EVALUATION VIEW
        # ═══════════════════════════════════════════════════════════════════
        evaluation_view = gr.Group(visible=False)
        with evaluation_view:
            gr.Markdown("## Evaluation and Model Transparency")
            gr.Markdown(
                "Performance metrics and model documentation for the Radiology Trust Layer. "
                "All metrics are computed on a held-out test set of 50 synthetic radiology cases."
            )
            with gr.Tabs():
                with gr.Tab("Before / After Metrics"):
                    gr.Markdown(
                        "Comparison of base MedGemma-4B-IT against the same model with the RTL LoRA adapter applied. "
                        "The LoRA adapter was trained on synthetic radiology data to improve structured output quality."
                    )
                    eval_metrics_html = gr.HTML(_render_default_metrics())
                with gr.Tab("Model Card"):
                    gr.Markdown(f"""
### Base Model: MedGemma 4B IT

MedGemma is a multimodal medical AI model developed by Google Health AI. It combines a Gemma 2 language model with a SigLIP vision encoder, enabling joint understanding of medical images and text.

| Property | Value |
|---|---|
| Base model | `{config.MEDGEMMA_MODEL_ID}` |
| Architecture | Gemma 2 + SigLIP vision encoder |
| Parameters | ~4 billion |
| Training data | Medical imaging and clinical text (Google Health AI) |
| Input modality | Image + Text (multimodal) |
| Access | Gated model on Hugging Face (requires agreement) |

### RTL LoRA Adapter

A lightweight Low-Rank Adaptation (LoRA) fine-tuned on top of MedGemma to improve two key behaviors:

1. **JSON schema compliance** -- ensures structured pipeline outputs are valid and parseable
2. **Uncertainty calibration** -- reduces overconfident language in generated text

| Property | Value |
|---|---|
| Adapter type | LoRA (PEFT) |
| Rank | r=4 |
| Target modules | q_proj, v_proj |
| Training | 8-bit quantized, SFTTrainer (TRL) on Kaggle T4 GPU |
| Dataset | 50 synthetic radiology cases |

**Adapter weights:** [View on Hugging Face](https://huggingface.co/outlawpink/rtl-medgemma-lora)
""")
                with gr.Tab("Example Cases"):
                    eval_case_md = gr.Markdown(_load_mock_example_md())
            evaluation_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # SETTINGS VIEW
        # ═══════════════════════════════════════════════════════════════════
        settings_view = gr.Group(visible=False)
        with settings_view:
            gr.Markdown("## System Configuration")
            gr.Markdown(
                "Current runtime configuration for the Radiology Trust Layer. "
                "These values are set at deployment time and control how the audit pipeline operates."
            )
            gr.Markdown(f"""
| Setting | Value | Description |
|---|---|---|
| Base model | `{config.MEDGEMMA_MODEL_ID}` | Google's medical vision-language model used for all inference |
| LoRA adapter | `{config.RTL_LORA_ID or 'Not loaded'}` | Custom fine-tuned adapter for JSON compliance and uncertainty calibration |
| Mock mode | `{'Enabled' if config.MEDGEMMA_MOCK else 'Disabled'}` | When enabled, returns pre-built results without running the model |
| Inference mode | `{config.MEDGEMMA_INFERENCE_MODE}` | How the model is loaded (local GPU, API, or mock) |
| Prompt version | `{config.RTL_PROMPT_VERSION}` | Version of the structured prompt templates used in the pipeline |
""")
            gr.Markdown("""
### About This System

RTL is a **research demonstration** built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle. It is designed to audit radiology report language against imaging evidence, not to generate diagnoses or replace clinical judgment.

**Important:** Do not upload real patient data. Always consult qualified radiologists for medical decisions.
""")
            settings_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # ALL VIEWS list (must match PAGES order)
        # ═══════════════════════════════════════════════════════════════════
        all_views = [
            landing_view, demo_view, login_view, home_view, single_view, batch_view,
            detail_view, history_view, evaluation_view, settings_view,
        ]

        # ═══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════════════════════════════

        def do_login(email: str, pw: str, st: dict):
            conn = get_conn(DB_PATH)
            uid = authenticate_user(conn, email.strip().lower(), pw)
            if not uid:
                return (st, _alert("Invalid credentials", "error")) + _set_views("login") + ("", []) + ("", "", "", "", "")
            st["user_id"] = uid
            st["page"] = "single"
            return (st, _alert("Logged in successfully", "success")) + _set_views("single") + ("", []) + ("", "", "", "", "")

        def do_create(email: str, name: str, pw: str, st: dict):
            try:
                conn = get_conn(DB_PATH)
                uid = create_user(conn, email.strip().lower(), name.strip(), pw)
            except Exception as e:
                return (st, _alert(str(e), "error")) + _set_views("login") + ("", []) + ("", "", "", "", "")
            st["user_id"] = uid
            st["page"] = "single"
            return (st, _alert("Account created", "success")) + _set_views("single") + ("", []) + ("", "", "", "", "")

        def go_to(page: str, st: dict):
            st["page"] = page
            header, recent = ("", [])
            if page == "home" and st.get("user_id"):
                header, recent = _after_login_data(st)
            return (st,) + _set_views(page) + (header, recent)

        def open_detail(run_id: str, st: dict):
            st["run_id"] = run_id.strip()
            st["page"] = "detail"
            detail_vals = _load_detail(run_id.strip())
            return (st,) + _set_views("detail") + detail_vals

        def run_single_audit(image, case_label: str, report: str, use_lora: bool, st: dict):
            if image is None or not report.strip():
                return (st, _alert("Please upload an image and paste a report", "error")) + _single_empty()

            try:
                from core.pipeline.audit_pipeline import run_audit

                result = run_audit(
                    image=image,
                    report_text=report,
                    case_label=case_label or "Untitled",
                    lora_id=config.RTL_LORA_ID if use_lora else "",
                )

                # Persist to DB only if logged in
                run_id = result["run_id"]
                if st.get("user_id"):
                    conn = get_conn(DB_PATH)
                    run_id = create_run(
                        conn,
                        user_id=st["user_id"],
                        image_hash=result["image_hash"],
                        report_hash=result["report_hash"],
                        case_label=result["case_label"],
                        model_name=result["model_name"],
                        model_version=result["model_version"],
                        lora_id=result.get("lora_id", ""),
                        prompt_version=result["prompt_version"],
                        overall_score=result["overall_score"],
                        severity=result["severity"],
                        flag_counts=result["flag_counts"],
                        results_path=result.get("results_path", ""),
                    )
                    log_ev(conn, run_id, EventType.PIPELINE_COMPLETE, {"score": result["overall_score"]})
                st["run_id"] = run_id
                st["current_result"] = result

                score_html = score_gauge_html(result["overall_score"], result["severity"])
                flag_html = flag_counts_html(result["flag_counts"])
                report_html = render_highlighted_report(
                    result["original_report"], result["alignments"], result["claims"]
                )
                claims_html = claim_table_html(result["alignments"])
                rewrites_html = rewrite_suggestions_html(result["rewrites"])

                cs = result.get("clinician_summary", {})
                clinician_md = (
                    f"**Summary:** {cs.get('summary', '')}\n\n"
                    f"**Recommendation:** {cs.get('recommendation', '').replace('_', ' ').title()}\n\n"
                    + ("**Key Concerns:**\n" + "\n".join(f"- {c}" for c in cs.get("key_concerns", [])) if cs.get("key_concerns") else "")
                    + f"\n\n*{cs.get('confidence_note', '')}*"
                )

                pe = result.get("patient_explanation", {})
                patient_md = (
                    f"**Summary:**\n\n{pe.get('plain_language_summary', '')}\n\n"
                    f"**What was found:** {pe.get('what_was_found', '')}\n\n"
                    f"**What it means:** {pe.get('what_it_means', '')}\n\n"
                    f"**Next steps:** {pe.get('next_steps', '')}"
                )

                edited = result.get("edited_report", report)

                status = _alert(
                    f"Audit complete — Run ID: {run_id} — Score: {result['overall_score']}/100 — {result['severity']} severity",
                    "success"
                )

                return (st, status, score_html, flag_html, report_html,
                        claims_html, rewrites_html, clinician_md, patient_md, edited)

            except Exception as e:
                logger.exception("Audit failed")
                return (st, _alert(f"Audit failed: {e}", "error")) + _single_empty()

        def accept_all_rewrites(st: dict):
            result = st.get("current_result")
            if not result:
                return ""
            return result.get("edited_report", "")

        def run_batch_audit(zip_file, batch_label_str: str, use_lora: bool, st: dict):
            if zip_file is None:
                return st, _alert("Please upload a ZIP file", "error"), "", "", []

            try:
                from core.batch.runner import run_batch

                extract_dir = config.BATCHES_DIR / "tmp_extract"
                zip_path = Path(zip_file.name)

                batch_result = run_batch(zip_path, extract_dir)
                results = batch_result["results"]
                summary = batch_result["summary"]

                # Persist to DB only if logged in
                if st.get("user_id"):
                    conn = get_conn(DB_PATH)
                    batch_id = create_batch(conn, user_id=st["user_id"],
                                            zip_name=zip_path.name, num_cases_total=0)
                    for r in results:
                        run_id = create_run(
                            conn, user_id=st["user_id"],
                            image_hash=r["image_hash"], report_hash=r["report_hash"],
                            case_label=r["case_label"], model_name=r["model_name"],
                            model_version=r["model_version"], lora_id=r.get("lora_id", ""),
                            prompt_version=r["prompt_version"], overall_score=r["overall_score"],
                            severity=r["severity"], flag_counts=r["flag_counts"],
                            results_path=r.get("results_path", ""),
                        )
                        link_batch_run(conn, batch_id, run_id, r["case_label"])
                    update_batch_progress(conn, batch_id,
                        num_done=summary["completed"], num_failed=summary["failed"],
                        summary=summary, status="complete")

                summary_md = (
                    f"**{summary['total_cases']} cases** — "
                    f"{summary['completed']} complete, {summary['failed']} failed\n\n"
                    f"**Average score:** {summary['avg_score']}/100  "
                    f"**Needs review:** {summary['pct_needing_review']}%\n\n"
                    f"Severity: {summary['severity_distribution']['low']} low, "
                    f"{summary['severity_distribution']['medium']} medium, "
                    f"{summary['severity_distribution']['high']} high"
                )

                table_rows = []
                for r in results:
                    fc = r["flag_counts"]
                    table_rows.append([
                        r["case_label"], r["overall_score"], r["severity"],
                        fc.get("supported", 0), fc.get("uncertain", 0), fc.get("needs_review", 0),
                        r["run_id"]
                    ])

                return (st,
                        _alert(f"Batch complete — {summary['completed']}/{summary['total_cases']} cases", "success"),
                        summary_md, "", table_rows)

            except Exception as e:
                logger.exception("Batch audit failed")
                return st, _alert(f"Batch failed: {e}", "error"), "", "", []

        def load_history(severity_filter: str, min_score: int, st: dict):
            if not st.get("user_id"):
                return gr.update(visible=True), gr.update(visible=False, value=[])
            conn = get_conn(DB_PATH)
            runs = list_all_runs_for_user(conn, st["user_id"])
            rows = []
            for r in runs:
                if severity_filter != "All" and r["severity"] != severity_filter:
                    continue
                if r["overall_score"] < min_score:
                    continue
                rows.append([
                    r["created_at"], r["case_label"], r["overall_score"],
                    r["severity"], r["model_version"], r["run_id"]
                ])
            return gr.update(visible=False), gr.update(visible=True, value=rows)

        def export_run(st: dict):
            result = st.get("current_result")
            if not result:
                run_id = st.get("run_id", "")
                if run_id:
                    p = config.RUNS_DIR / run_id / "results.json"
                    if p.exists():
                        return gr.update(visible=True, value=str(p))
            if result:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
                json.dump(result, tmp, indent=2)
                tmp.close()
                return gr.update(visible=True, value=tmp.name)
            return gr.update(visible=False)

        # ═══════════════════════════════════════════════════════════════════
        # WIRE UP BUTTONS
        # ═══════════════════════════════════════════════════════════════════

        # Shared output list: state + nav_group + all page views + home_header + recent_table
        _nav_buttons = [nav_home_btn, nav_demo, nav_single, nav_batch, nav_history, nav_eval, nav_settings]
        _shared_nav_outputs = [state, nav_group, nav_page_info] + _nav_buttons + all_views + [home_header, recent_table]

        # Detail component list
        _detail_comps = [
            detail_run_md, detail_score_html, detail_flag_html, detail_meta_md,
            detail_report_html, detail_claims_html, detail_rewrites_html,
            detail_clinician_md, detail_patient_md, detail_edited_txt, detail_trail_html,
        ]

        def _detail_empty():
            return ("",) * len(_detail_comps)

        def _load_detail(run_id: str):
            try:
                p = config.RUNS_DIR / run_id / "results.json"
                if not p.exists():
                    return ("Run not found",) + ("",) * (len(_detail_comps) - 1)
                result = read_json(p)

                conn = get_conn(DB_PATH)
                events = list_events_for_run(conn, run_id)

                run_md = f"**Run ID:** `{run_id}` | **Created:** {result.get('created_at', '')} | **Case:** {result.get('case_label', '')}"
                score_h = score_gauge_html(result["overall_score"], result["severity"])
                flag_h = flag_counts_html(result["flag_counts"])
                meta = (
                    f"**Model:** `{result.get('model_version', '')}`\n\n"
                    f"**Prompt version:** `{result.get('prompt_version', '')}`\n\n"
                    f"**Image quality:** {result.get('image_quality', '')}\n\n"
                    f"**Mock mode:** {'Yes' if result.get('mock_mode') else 'No'}"
                )
                report_h = render_highlighted_report(
                    result.get("original_report", ""), result.get("alignments", []), result.get("claims", [])
                )
                claims_h = claim_table_html(result.get("alignments", []))
                rewrites_h = rewrite_suggestions_html(result.get("rewrites", []))

                cs = result.get("clinician_summary", {})
                clinician = (
                    f"**{cs.get('summary', '')}**\n\n"
                    f"Recommendation: {cs.get('recommendation', '').replace('_', ' ').title()}\n\n"
                    + "\n".join(f"- {c}" for c in cs.get("key_concerns", []))
                )
                pe = result.get("patient_explanation", {})
                patient = pe.get("plain_language_summary", "")
                edited = result.get("edited_report", "")

                trail_rows = ""
                for ev in events:
                    trail_rows += (
                        f"<tr><td class='rtl-td' style='font-size:0.8rem;'>{ev['timestamp']}</td>"
                        f"<td class='rtl-td' style='font-size:0.8rem;'>{ev['event_type']}</td>"
                        f"<td class='rtl-td' style='font-size:0.8rem;color:#5f6368;'>{ev['actor']}</td></tr>"
                    )
                if trail_rows:
                    trail_html = (
                        "<table class='rtl-table'>"
                        "<thead><tr><th class='rtl-th'>Time</th>"
                        "<th class='rtl-th'>Event</th>"
                        "<th class='rtl-th'>Actor</th></tr></thead>"
                        "<tbody>" + trail_rows + "</tbody></table>"
                    )
                else:
                    trail_html = "<p style='color:#5f6368;'>No events logged.</p>"

                return (run_md, score_h, flag_h, meta, report_h, claims_h, rewrites_h,
                        clinician, patient, edited, trail_html)
            except Exception as e:
                return (f"Error loading run: {e}",) + ("",) * (len(_detail_comps) - 1)

        # Login / Create
        _login_form_fields = [login_email, login_pw, create_email, create_name_box, create_pw]
        login_btn.click(
            do_login,
            inputs=[login_email, login_pw, state],
            outputs=[state, login_msg, nav_group, nav_page_info] + _nav_buttons + all_views + [home_header, recent_table] + _login_form_fields,
        )
        create_btn.click(
            do_create,
            inputs=[create_email, create_name_box, create_pw, state],
            outputs=[state, create_msg, nav_group, nav_page_info] + _nav_buttons + all_views + [home_header, recent_table] + _login_form_fields,
        )

        # Open detail from home / history
        home_open_btn.click(
            lambda run_id, st: (st,) + _set_views("detail") + _load_detail(run_id.strip()),
            inputs=[home_run_id_box, state],
            outputs=[state, nav_group, nav_page_info] + _nav_buttons + all_views + _detail_comps,
        )
        hist_open_btn.click(
            lambda run_id, st: (st,) + _set_views("detail") + _load_detail(run_id.strip()),
            inputs=[hist_run_id_box, state],
            outputs=[state, nav_group, nav_page_info] + _nav_buttons + all_views + _detail_comps,
        )

        # Navigation — persistent nav bar
        nav_home_btn.click(lambda st: go_to("landing", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_demo.click(lambda st: go_to("demo", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_single.click(lambda st: go_to("single", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_batch.click(lambda st: go_to("batch", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_history.click(
            lambda st: go_to("history", st), inputs=[state], outputs=_shared_nav_outputs
        ).then(
            load_history,
            inputs=[hist_filter_sev, hist_filter_score, state],
            outputs=[hist_placeholder, hist_table],
        )
        nav_eval.click(lambda st: go_to("evaluation", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_settings.click(lambda st: go_to("settings", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_login.click(lambda st: go_to("login", st), inputs=[state], outputs=_shared_nav_outputs)

        # Single audit — show loading in results area, then run
        single_run_btn.click(
            lambda: (_loading_results_html(1, 6), "", "", "", "", "", "", ""),
            outputs=[single_score_html, single_flag_html, single_report_html,
                     single_claims_html, single_rewrites_html, single_clinician_md,
                     single_patient_md, single_edited_report],
        ).then(
            run_single_audit,
            inputs=[single_image, single_case_label, single_report, single_lora, state],
            outputs=[
                state, single_status, single_score_html, single_flag_html,
                single_report_html, single_claims_html, single_rewrites_html,
                single_clinician_md, single_patient_md, single_edited_report,
            ],
        )
        single_accept_all.click(accept_all_rewrites, inputs=[state], outputs=[single_edited_report])

        # Demo — card clicks load image+report, Run Audit shows loading then results
        demo_btn_1.click(lambda: _load_example_case(0), outputs=[demo_image, demo_case_label, demo_report])
        demo_btn_2.click(lambda: _load_example_case(1), outputs=[demo_image, demo_case_label, demo_report])
        demo_btn_3.click(lambda: _load_example_case(2), outputs=[demo_image, demo_case_label, demo_report])

        def _run_demo_with_loading(image, case_label, report):
            """Simulate audit with preloaded results and brief loading delay."""
            import time
            if image is None or not report.strip():
                return (_alert("Select a case above first", "error"),) + ("",) * 6
            # Find which case this is
            idx = 0
            for i, ex in enumerate(EXAMPLE_CASES):
                if ex["label"] == case_label:
                    idx = i
                    break
            time.sleep(1.5)  # brief simulated delay
            result = _load_preloaded_demo(idx)
            # result = (image, label, report, score, flag, report_html, claims, rewrites, clinician, patient)
            return result[3:]  # skip image/label/report, return score through patient

        _demo_result_outputs = [
            demo_score_html, demo_flag_html, demo_report_html, demo_claims_html,
            demo_rewrites_html, demo_clinician_md, demo_patient_md,
        ]
        demo_run_btn.click(
            lambda: (_loading_results_html(1, 6), "", "", "", "", "", ""),
            outputs=_demo_result_outputs,
        ).then(
            _run_demo_with_loading,
            inputs=[demo_image, demo_case_label, demo_report],
            outputs=_demo_result_outputs,
        )

        # Batch audit — show loading spinner, then run
        batch_run_btn.click(
            lambda: _loading_html("Running batch audit — processing cases..."),
            outputs=[batch_status],
        ).then(
            run_batch_audit,
            inputs=[batch_zip, batch_label_box, batch_lora, state],
            outputs=[state, batch_status, batch_summary_md, batch_progress_md, batch_table],
        )

        # History
        hist_refresh.click(
            load_history,
            inputs=[hist_filter_sev, hist_filter_score, state],
            outputs=[hist_placeholder, hist_table],
        )

        # Export
        detail_export_btn.click(export_run, inputs=[state], outputs=[detail_export_file])

        # Back buttons (return to single audit; login back returns to landing)
        back_buttons = [single_back, batch_back, detail_back, history_back, evaluation_back, settings_back]
        for btn in back_buttons:
            btn.click(lambda st: go_to("single", st), inputs=[state], outputs=_shared_nav_outputs)
        login_back.click(lambda st: go_to("landing", st), inputs=[state], outputs=_shared_nav_outputs)

        # Initial view on load
        demo.load(
            lambda st: (st,) + _set_views(st.get("page", "landing")) + ("", []),
            inputs=[state],
            outputs=[state, nav_group, nav_page_info] + _nav_buttons + all_views + [home_header, recent_table],
        )

    return demo


# ─────────────────────────── Standalone helpers ───────────────────────────

def _single_empty():
    return ("",) * 8  # score, flag, report, claims, rewrites, clinician, patient, edited


def _render_default_metrics() -> str:
    """Render the before/after evaluation metrics table with real training results."""
    rows = [
        ("JSON Schema Valid Rate", "84.0%", "100.0%", "+16.0%", True),
        ("Overconfidence Rate", "10.0%", "0.0%", "-10.0%", True),
        ("Label Value Valid Rate", "80.0%", "100.0%", "+20.0%", True),
        ("Label Accuracy", "65.3%", "87.3%", "+22.0%", True),
        ("Schema Repair Needed Rate", "84.0%", "0.0%", "-84.0%", True),
    ]
    html_rows = ""
    for metric, base, lora, delta, improved in rows:
        color = "#137333" if improved else "#c5221f"
        html_rows += (
            f"<tr>"
            f"<td class='rtl-td'>{metric}</td>"
            f"<td class='rtl-td' style='text-align:center;'>{base}</td>"
            f"<td class='rtl-td' style='text-align:center;font-weight:600;'>{lora}</td>"
            f"<td class='rtl-td' style='text-align:center;color:{color};font-weight:600;'>{delta}</td>"
            f"</tr>"
        )
    return (
        "<p style='color:#5f6368;font-size:0.85rem;margin-bottom:12px;'>"
        "Evaluated on 50 synthetic radiology cases. "
        "Base model: MedGemma-4B-IT (8-bit quantized). "
        "LoRA adapter: RTL-v1 (r=4, target modules: q_proj, v_proj).</p>"
        "<table class='rtl-table'>"
        "<thead><tr>"
        "<th class='rtl-th'>Metric</th>"
        "<th class='rtl-th' style='text-align:center;'>Base MedGemma</th>"
        "<th class='rtl-th' style='text-align:center;'>+ RTL LoRA</th>"
        "<th class='rtl-th' style='text-align:center;'>Delta</th>"
        "</tr></thead><tbody>" + html_rows + "</tbody></table>"
    )


def _load_mock_example_md() -> str:
    try:
        p = config.MOCK_RESULTS_PATH
        if p.exists():
            data = json.loads(p.read_text())
            case = data if isinstance(data, dict) else {}
            score = case.get("overall_score", "?")
            sev = case.get("severity", "?")
            label = case.get("case_label", "Example Case")
            claims = case.get("alignments", [])
            claim_lines = "\n".join(
                f"- {a.get('label', '?').upper()}: {a.get('claim_text', a.get('claim_id', ''))}"
                for a in claims[:5]
            )
            return (
                f"### {label}\n"
                f"**Score:** {score}/100  **Severity:** {sev}\n\n"
                f"**Claims:**\n{claim_lines}"
            )
    except Exception:
        pass
    return "### Example Case\nLoad mock data to see a worked example."


if __name__ == "__main__":
    app = main()
    app.launch()
