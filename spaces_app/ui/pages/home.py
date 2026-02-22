"""Home page — dashboard tiles and recent run history."""
import gradio as gr


def build(nav_cb):
    """Build home page components. Returns dict of key components."""
    header = gr.Markdown("# Radiology Trust Layer")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("""
### Welcome to RTL — Radiology Trust Layer

RTL uses **MedGemma** (Google's open-weight medical AI) to audit radiology reports
against imaging evidence. It surfaces where language is well-supported, uncertain,
or potentially misleading — without generating diagnoses.

**This is a public demo. Do not upload PHI or real patient data.**
""")
        with gr.Column(scale=1):
            gr.Markdown("""
**Quick start**
1. Click **Single Study Audit**
2. Upload a chest X-ray image
3. Paste a radiology report
4. Click **Run Audit**
""")

    with gr.Row():
        go_single = gr.Button("🔬 Single Study Audit", variant="primary", size="lg")
        go_batch = gr.Button("📦 Batch Audit", size="lg")
        go_history = gr.Button("📋 History", size="lg")
        go_eval = gr.Button("📊 Evaluation", size="lg")

    gr.Markdown("---\n### Recent Audits")
    recent_table = gr.Dataframe(
        headers=["Date", "Case Label", "Score", "Severity", "Run ID"],
        datatype=["str", "str", "number", "str", "str"],
        interactive=False,
        wrap=True,
    )

    with gr.Row():
        open_run_id = gr.Textbox(label="Open run by ID", placeholder="Paste a run_id to open Audit Detail")
        open_btn = gr.Button("Open Audit Detail", size="sm")

    return {
        "header": header,
        "go_single": go_single,
        "go_batch": go_batch,
        "go_history": go_history,
        "go_eval": go_eval,
        "recent_table": recent_table,
        "open_run_id": open_run_id,
        "open_btn": open_btn,
    }
