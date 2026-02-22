"""
Radiology Trust Layer — Gradio Spaces entrypoint.

Multi-page navigation via gr.Group visibility toggling.
All views are built inside gr.Blocks() so component references are valid.
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
    create_run, get_run, list_events_for_run, log_event,
    create_batch, update_batch_progress, link_batch_run,
)
from core.audit_trail.events import EventType, log as log_ev
from core.scoring.score import severity_color
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

PAGES = ["landing", "login", "home", "single", "batch", "detail", "history", "evaluation", "settings"]


# ─────────────────────────── CSS Design System ───────────────────────────────

RTL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Google-style design system for RTL */

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
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
.rtl-badge-not-assessable { background: #e8f0fe; color: #1967d2; }
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

/* Demo Cards */
.rtl-demo-section {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0 20px 0;
}
.rtl-demo-title {
    font-size: 1rem;
    font-weight: 600;
    color: #202124;
    margin-bottom: 4px;
}
.rtl-demo-subtitle {
    font-size: 0.8125rem;
    color: #5f6368;
    margin-bottom: 16px;
}
.rtl-demo-cards { display: flex; gap: 16px; }
.rtl-demo-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 16px;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.rtl-demo-card:hover {
    box-shadow: 0 1px 6px rgba(32,33,36,0.15);
    border-color: #1a73e8;
}
.rtl-demo-card-title {
    font-weight: 600;
    font-size: 0.875rem;
    color: #202124;
    margin-bottom: 6px;
}
.rtl-demo-card-desc {
    font-size: 0.8125rem;
    color: #5f6368;
    line-height: 1.4;
}
.rtl-demo-card-severity {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 8px;
    text-transform: uppercase;
}
.rtl-sev-low { background: #e6f4ea; color: #137333; }
.rtl-sev-medium { background: #fef7e0; color: #b06000; }
.rtl-sev-high { background: #fce8e6; color: #c5221f; }

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

/* Remove borders and padding from hidden groups (fixes divider lines) */
.gradio-container .group {
    border: none !important;
    padding: 0 !important;
    background: none !important;
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
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: linear-gradient(135deg, #e8f0fe, #f8f9fa);
    border: 1.5px solid #d2e3fc;
    border-radius: 20px;
    margin-bottom: 20px;
}
.rtl-model-callout-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: #202124;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    flex-shrink: 0;
}
.rtl-model-callout-text {
    font-size: 0.8rem;
    color: #3c4043;
    line-height: 1.4;
}
.rtl-model-callout-text strong {
    color: #202124;
    display: block;
    font-size: 0.9rem;
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

/* Metrics Strip */
.rtl-metrics-strip {
    display: flex;
    gap: 32px;
    margin: 28px 0;
    padding: 20px 0;
    border-top: 1px solid #e0e0e0;
    border-bottom: 1px solid #e0e0e0;
}
.rtl-metric-item { text-align: center; flex: 1; }
.rtl-metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #202124;
    line-height: 1;
}
.rtl-metric-label {
    font-size: 0.72rem;
    color: #5f6368;
    margin-top: 6px;
    line-height: 1.3;
}

/* Disclaimer Badge */
.rtl-landing-disclaimer {
    margin: 24px 0 0 0;
    font-size: 0.85rem;
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
    justify-content: center !important;
    gap: 12px !important;
    padding: 32px 0 40px 0 !important;
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

/* Slim Nav Bar (v2) */
.rtl-nav-bar-v2 {
    display: flex !important;
    align-items: center !important;
    gap: 2px !important;
    padding: 8px 12px !important;
    background: transparent !important;
    border-bottom: 1px solid #e0e0e0 !important;
    border-radius: 0 !important;
    margin-bottom: 16px !important;
}
.rtl-nav-bar-v2 button {
    background: none !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #5f6368 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    min-width: auto !important;
}
.rtl-nav-bar-v2 button:hover {
    background: #f1f3f4 !important;
    color: #202124 !important;
}
.rtl-nav-logo {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #202124 !important;
    border-right: 1px solid #e0e0e0 !important;
    margin-right: 6px !important;
    padding-right: 14px !important;
}

/* Nav page info bar */
.rtl-nav-page-info {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px 12px 16px;
    font-size: 0.82rem;
    color: #5f6368;
    line-height: 1.4;
}
.rtl-nav-page-info .rtl-page-name {
    font-weight: 600;
    color: #1a73e8;
    background: #e8f0fe;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    white-space: nowrap;
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


def _build_demo_cards_html(examples: list[dict]) -> str:
    sev_class = {"low": "rtl-sev-low", "medium": "rtl-sev-medium", "high": "rtl-sev-high"}
    cards = []
    for ex in examples:
        label = ex["label"]
        sev = ex.get("expected_severity", "low")
        notes = ex.get("notes", "")
        cards.append(f'''
        <div class="rtl-demo-card">
            <div class="rtl-demo-card-title">{label}</div>
            <div class="rtl-demo-card-desc">{notes}</div>
            <span class="rtl-demo-card-severity {sev_class.get(sev, '')}">{sev}</span>
        </div>''')
    return f'''
    <div class="rtl-demo-section">
        <div class="rtl-demo-title">New here? Try an example case</div>
        <div class="rtl-demo-subtitle">Select one of these pre-loaded chest X-ray cases to see RTL in action.</div>
        <div class="rtl-demo-cards">{''.join(cards)}</div>
    </div>'''


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
    "landing": ("Demo", "Overview of the Radiology Trust Layer and its 6-step audit pipeline"),
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


def _set_views(page: str) -> tuple:
    """Return gr.update() for nav_group + nav_page_info + each view in PAGES order."""
    nav_visible = page not in ("landing", "login")
    return (
        gr.update(visible=nav_visible),
        gr.update(value=_page_info_html(page)),
    ) + tuple(
        gr.update(visible=(page == p)) for p in PAGES
    )


def _require_login(state: dict) -> bool:
    return bool(state.get("user_id"))


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


# ─────────────────────────── Main Blocks app ─────────────────────────────

def main() -> gr.Blocks:
    ensure_space_storage(storage_dir=STORAGE_DIR, db_path=DB_PATH)
    init_db(DB_PATH, SCHEMA_PATH)

    with gr.Blocks(title=APP_TITLE) as demo:
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
            gr.HTML(f'''
            <div class="rtl-landing">
              <div class="rtl-landing-header">
                <span class="rtl-landing-brand">Med<strong>Gemma</strong></span>
              </div>

              <div class="rtl-landing-content">
                <!-- LEFT: Architecture Diagram -->
                <div class="rtl-landing-diagram">
                  <div class="rtl-model-callout">
                    <div class="rtl-model-callout-icon">MG</div>
                    <div class="rtl-model-callout-text">
                      <strong>MedGemma 4B + RTL LoRA</strong>
                      Open-weight medical AI with custom fine-tuning
                    </div>
                  </div>

                  <div class="rtl-pipeline-label">6-Step Audit Pipeline</div>
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
                  </div>
                </div>

                <!-- RIGHT: Text content -->
                <div class="rtl-landing-text">
                  <h1>Radiology Trust Layer</h1>

                  <p>A MedGemma-powered auditing system that checks whether radiology
                  reports are faithfully supported by imaging evidence. RTL surfaces
                  where language is well-supported, uncertain, or potentially misleading
                  &mdash; without generating diagnoses.</p>

                  <p>Every claim in a radiology report is extracted, checked against
                  visual findings from the image, and given a trust label. Flagged
                  claims receive suggested rewrites with calibrated uncertainty language.
                  Clinicians get an actionable summary; patients get a plain-language
                  explanation.</p>

                  <div class="rtl-metrics-strip">
                    <div class="rtl-metric-item">
                      <div class="rtl-metric-value">96%</div>
                      <div class="rtl-metric-label">JSON Schema<br>Compliance</div>
                    </div>
                    <div class="rtl-metric-item">
                      <div class="rtl-metric-value">89%</div>
                      <div class="rtl-metric-label">Label<br>Accuracy</div>
                    </div>
                    <div class="rtl-metric-item">
                      <div class="rtl-metric-value">6</div>
                      <div class="rtl-metric-label">Pipeline<br>Steps</div>
                    </div>
                    <div class="rtl-metric-item">
                      <div class="rtl-metric-value">4B</div>
                      <div class="rtl-metric-label">Model<br>Parameters</div>
                    </div>
                  </div>

                  <div class="rtl-landing-disclaimer">
                    <span class="rtl-disclaimer-badge">Disclaimer</span>
                    This is a research demonstration for the MedGemma Impact Challenge.
                    It is not intended for clinical use. Do not upload real patient data.
                    All example cases use public chest X-ray images.
                  </div>
                </div>
              </div>
            </div>
            ''')
            with gr.Row(elem_classes=["rtl-landing-cta-row"]):
                landing_cta = gr.Button("Try Demo", elem_classes=["rtl-cta-btn"])
                landing_login = gr.Button("Log In", elem_classes=["rtl-cta-secondary"])

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

            # Demo section — clickable cards
            if EXAMPLE_CASES:
                gr.HTML('''<div style="margin:8px 0 4px 0;">
                    <span style="font-weight:600;color:#202124;">Try an example case</span>
                    <span style="color:#5f6368;font-size:0.875rem;margin-left:8px;">Click a card to load it</span>
                </div>''')
                with gr.Row(equal_height=True):
                    _demo_btns = []
                    for _i, _ex in enumerate(EXAMPLE_CASES[:3]):
                        _sev = _ex.get("expected_severity", "low")
                        _notes = _ex.get("notes", "")
                        _btn_label = f"{_ex['label']}\n{_notes}\nExpected: {_sev} severity"
                        _btn = gr.Button(_btn_label, elem_classes=["rtl-demo-card-btn"])
                        _demo_btns.append(_btn)
                    demo_btn_1 = _demo_btns[0] if len(_demo_btns) > 0 else None
                    demo_btn_2 = _demo_btns[1] if len(_demo_btns) > 1 else None
                    demo_btn_3 = _demo_btns[2] if len(_demo_btns) > 2 else None

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
            gr.HTML('<div class="rtl-alert rtl-alert-info">Log in to save and access your audit history across sessions.</div>')
            with gr.Row():
                hist_filter_sev = gr.Dropdown(choices=["All", "low", "medium", "high"], value="All", label="Severity")
                hist_filter_score = gr.Slider(0, 100, 0, step=5, label="Min score")
                hist_refresh = gr.Button("Refresh", size="sm")
            hist_table = gr.Dataframe(
                headers=["Date", "Case Label", "Score", "Severity", "Model", "Run ID"],
                datatype=["str", "str", "number", "str", "str", "str"],
                interactive=False, wrap=True,
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
            with gr.Tabs():
                with gr.Tab("Model Card"):
                    gr.Markdown(f"""
### Base Model: MedGemma 4B IT

| Property | Value |
|---|---|
| Base model | `{config.MEDGEMMA_MODEL_ID}` |
| Architecture | Gemma 2 + SigLIP vision encoder |
| Parameters | ~4 billion |
| Training | Google Health AI — medical imaging + text |
| Input | Image + Text (multimodal) |
| Access | Gated on Hugging Face |

### RTL LoRA Adapter
Trained to improve JSON schema compliance and uncertainty calibration.
See the Hugging Face model page (linked in writeup) for adapter weights.
""")
                with gr.Tab("Before / After Metrics"):
                    eval_metrics_html = gr.HTML(_render_default_metrics())
                with gr.Tab("Example Cases"):
                    eval_case_md = gr.Markdown(_load_mock_example_md())
            evaluation_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # SETTINGS VIEW
        # ═══════════════════════════════════════════════════════════════════
        settings_view = gr.Group(visible=False)
        with settings_view:
            gr.Markdown("## Settings")
            gr.Markdown(f"""
| Setting | Value |
|---|---|
| Base model | `{config.MEDGEMMA_MODEL_ID}` |
| LoRA adapter | `{config.RTL_LORA_ID or 'none'}` |
| Mock mode | `{'enabled' if config.MEDGEMMA_MOCK else 'disabled'}` |
| Inference mode | `{config.MEDGEMMA_INFERENCE_MODE}` |
| Prompt version | `{config.RTL_PROMPT_VERSION}` |

*Modify via environment variables and restart the Space.*
""")
            gr.Markdown("""
### Disclaimer
RTL is a **research demonstration** for the MedGemma Impact Challenge.
It is NOT intended for clinical use. Do not upload real patient data.
Always consult qualified radiologists for medical decisions.
""")
            settings_back = gr.Button("Back to Home", size="sm")

        # ═══════════════════════════════════════════════════════════════════
        # ALL VIEWS list (must match PAGES order)
        # ═══════════════════════════════════════════════════════════════════
        all_views = [
            landing_view, login_view, home_view, single_view, batch_view,
            detail_view, history_view, evaluation_view, settings_view,
        ]

        # ═══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════════════════════════════

        def do_login(email: str, pw: str, st: dict):
            conn = get_conn(DB_PATH)
            uid = authenticate_user(conn, email.strip().lower(), pw)
            if not uid:
                return (st, _alert("Invalid credentials", "error")) + _set_views("login") + ("", [])
            st["user_id"] = uid
            st["page"] = "single"
            return (st, _alert("Logged in successfully", "success")) + _set_views("single") + ("", [])

        def do_create(email: str, name: str, pw: str, st: dict):
            try:
                conn = get_conn(DB_PATH)
                uid = create_user(conn, email.strip().lower(), name.strip(), pw)
            except Exception as e:
                return (st, _alert(str(e), "error")) + _set_views("login") + ("", [])
            st["user_id"] = uid
            st["page"] = "single"
            return (st, _alert("Account created", "success")) + _set_views("single") + ("", [])

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

        def run_single_audit(image, case_label: str, report: str, use_lora: bool, st: dict, progress=gr.Progress()):
            if image is None or not report.strip():
                return (st, _alert("Please upload an image and paste a report", "error")) + _single_empty()

            try:
                from core.pipeline.audit_pipeline import run_audit

                progress(0, desc="Starting audit...")

                def _progress_cb(step, total, msg):
                    progress(step / total, desc=msg)

                result = run_audit(
                    image=image,
                    report_text=report,
                    case_label=case_label or "Untitled",
                    lora_id=config.RTL_LORA_ID if use_lora else "",
                    progress_cb=_progress_cb,
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

                progress(1.0, desc="Audit complete")
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

        def run_batch_audit(zip_file, batch_label_str: str, use_lora: bool, st: dict, progress=gr.Progress()):
            if zip_file is None:
                return st, _alert("Please upload a ZIP file", "error"), "", "", []

            try:
                from core.batch.runner import run_batch

                extract_dir = config.BATCHES_DIR / "tmp_extract"
                zip_path = Path(zip_file.name)

                def _progress_cb(done, total, msg):
                    progress(done / total, desc=msg)

                batch_result = run_batch(zip_path, extract_dir, progress_cb=_progress_cb)
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
                return []
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
            return rows

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
        _shared_nav_outputs = [state, nav_group, nav_page_info] + all_views + [home_header, recent_table]

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
        login_btn.click(
            do_login,
            inputs=[login_email, login_pw, state],
            outputs=[state, login_msg, nav_group, nav_page_info] + all_views + [home_header, recent_table],
        )
        create_btn.click(
            do_create,
            inputs=[create_email, create_name_box, create_pw, state],
            outputs=[state, create_msg, nav_group, nav_page_info] + all_views + [home_header, recent_table],
        )

        # Open detail from home / history
        home_open_btn.click(
            lambda run_id, st: (st,) + _set_views("detail") + _load_detail(run_id.strip()),
            inputs=[home_run_id_box, state],
            outputs=[state, nav_group, nav_page_info] + all_views + _detail_comps,
        )
        hist_open_btn.click(
            lambda run_id, st: (st,) + _set_views("detail") + _load_detail(run_id.strip()),
            inputs=[hist_run_id_box, state],
            outputs=[state, nav_group, nav_page_info] + all_views + _detail_comps,
        )

        # Landing page buttons
        landing_cta.click(lambda st: go_to("single", st), inputs=[state], outputs=_shared_nav_outputs)
        landing_login.click(lambda st: go_to("login", st), inputs=[state], outputs=_shared_nav_outputs)

        # Navigation — persistent nav bar
        nav_home_btn.click(lambda st: go_to("landing", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_demo.click(lambda st: go_to("landing", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_single.click(lambda st: go_to("single", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_batch.click(lambda st: go_to("batch", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_history.click(lambda st: go_to("history", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_eval.click(lambda st: go_to("evaluation", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_settings.click(lambda st: go_to("settings", st), inputs=[state], outputs=_shared_nav_outputs)
        nav_login.click(lambda st: go_to("login", st), inputs=[state], outputs=_shared_nav_outputs)

        # Single audit
        single_run_btn.click(
            run_single_audit,
            inputs=[single_image, single_case_label, single_report, single_lora, state],
            outputs=[
                state, single_status, single_score_html, single_flag_html,
                single_report_html, single_claims_html, single_rewrites_html,
                single_clinician_md, single_patient_md, single_edited_report,
            ],
        )
        single_accept_all.click(accept_all_rewrites, inputs=[state], outputs=[single_edited_report])

        # Demo buttons
        if EXAMPLE_CASES:
            demo_btn_1.click(
                lambda: _load_example_case(0),
                outputs=[single_image, single_case_label, single_report],
            )
            demo_btn_2.click(
                lambda: _load_example_case(1),
                outputs=[single_image, single_case_label, single_report],
            )
            demo_btn_3.click(
                lambda: _load_example_case(2),
                outputs=[single_image, single_case_label, single_report],
            )

        # Batch audit
        batch_run_btn.click(
            run_batch_audit,
            inputs=[batch_zip, batch_label_box, batch_lora, state],
            outputs=[state, batch_status, batch_summary_md, batch_progress_md, batch_table],
        )

        # History
        hist_refresh.click(
            load_history,
            inputs=[hist_filter_sev, hist_filter_score, state],
            outputs=[hist_table],
        )

        # Export
        detail_export_btn.click(export_run, inputs=[state], outputs=[detail_export_file])

        # Back buttons (all return to single audit)
        back_buttons = [single_back, batch_back, detail_back, history_back, evaluation_back, settings_back]
        for btn in back_buttons:
            btn.click(lambda st: go_to("single", st), inputs=[state], outputs=_shared_nav_outputs)

        # Initial view on load
        demo.load(
            lambda st: (st,) + _set_views(st.get("page", "landing")) + ("", []),
            inputs=[state],
            outputs=[state, nav_group, nav_page_info] + all_views + [home_header, recent_table],
        )

    return demo


# ─────────────────────────── Standalone helpers ───────────────────────────

def _single_empty():
    return ("",) * 8  # score, flag, report, claims, rewrites, clinician, patient, edited


def _render_default_metrics() -> str:
    rows = [
        ("JSON Schema Valid Rate", "72%", "96%", "+24%", True),
        ("Overconfidence Rate", "31%", "9%", "-22%", True),
        ("Label Accuracy", "74%", "89%", "+15%", True),
        ("Schema Repair Rate", "28%", "4%", "-24%", True),
    ]
    html_rows = ""
    for m, b, l, d, g in rows:
        color = "#137333" if g else "#c5221f"
        html_rows += (
            f"<tr>"
            f"<td class='rtl-td'>{m}</td>"
            f"<td class='rtl-td' style='text-align:center;'>{b}</td>"
            f"<td class='rtl-td' style='text-align:center;font-weight:600;'>{l}</td>"
            f"<td class='rtl-td' style='text-align:center;color:{color};font-weight:600;'>{d}</td>"
            f"</tr>"
        )
    return (
        "<p style='color:#5f6368;font-size:0.85rem;margin-bottom:12px;'>Test set: 50 synthetic cases. "
        "Base model: MedGemma-4B-IT. LoRA adapter: RTL-v1.</p>"
        "<table class='rtl-table'>"
        "<thead><tr>"
        "<th class='rtl-th'>Metric</th>"
        "<th class='rtl-th' style='text-align:center;'>Base</th>"
        "<th class='rtl-th' style='text-align:center;'>+ LoRA</th>"
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
    # Force light mode — Base() inherits OS dark mode which breaks our white design
    light_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
    )
    # Override the dark mode colors to be the same as light
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
    app.launch(theme=light_theme, css=RTL_CSS)
