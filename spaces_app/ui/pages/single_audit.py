"""Single Study Audit page — upload image + report, run pipeline, show results."""
import gradio as gr
from pathlib import Path

from spaces_app.ui.components import score_gauge_html, flag_counts_html, claim_table_html, rewrite_suggestions_html
from spaces_app.ui.render_report import render_highlighted_report
from core import config

EXAMPLES_DIR = config.EXAMPLES_DIR


def build():
    """Build the single audit page. Returns dict of key components."""

    gr.Markdown("## 🔬 Single Study Audit")
    gr.Markdown(
        "Upload a radiology image and paste the corresponding report. "
        "RTL will audit each claim against visual evidence using MedGemma."
    )

    with gr.Row():
        # Left: inputs
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            image_input = gr.Image(
                label="Radiology Image",
                type="pil",
                height=320,
            )
            case_label_input = gr.Textbox(label="Case label (optional)", placeholder="e.g. CXR_2024_001")
            report_input = gr.Textbox(
                label="Radiology Report Text",
                lines=10,
                placeholder="Paste the free-text radiology report here...",
            )

            # Example loader
            example_files = _get_example_files()
            if example_files:
                with gr.Accordion("Load an example case", open=False):
                    example_dropdown = gr.Dropdown(
                        choices=[e["label"] for e in example_files],
                        label="Select example",
                        interactive=True,
                    )
                    load_example_btn = gr.Button("Load Example", size="sm")
            else:
                example_dropdown = gr.Dropdown(choices=[], label="No examples found", visible=False)
                load_example_btn = gr.Button(visible=False)

            use_lora = gr.Checkbox(label="Use RTL LoRA adapter (improved JSON compliance)", value=False)
            run_btn = gr.Button("Run Audit", variant="primary", size="lg")
            status_md = gr.Markdown()

        # Right: results
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            score_html = gr.HTML("<p style='color:#9ca3af;'>Results will appear here after running the audit.</p>")
            flag_html = gr.HTML()

            with gr.Tabs():
                with gr.Tab("Report Highlights"):
                    report_html = gr.HTML()
                with gr.Tab("Claim Analysis"):
                    claims_html = gr.HTML()
                with gr.Tab("Suggested Rewrites"):
                    rewrites_html = gr.HTML()
                    accept_all_btn = gr.Button("Accept All Rewrites", size="sm")
                with gr.Tab("Clinician Summary"):
                    clinician_md = gr.Markdown()
                with gr.Tab("Patient Explanation"):
                    patient_md = gr.Markdown()
                with gr.Tab("Edited Report"):
                    edited_report = gr.Textbox(label="Edited Report (with accepted rewrites)", lines=10)

    back_btn = gr.Button("Back to Home")

    return {
        "image_input": image_input,
        "case_label_input": case_label_input,
        "report_input": report_input,
        "example_dropdown": example_dropdown,
        "load_example_btn": load_example_btn,
        "use_lora": use_lora,
        "run_btn": run_btn,
        "status_md": status_md,
        "score_html": score_html,
        "flag_html": flag_html,
        "report_html": report_html,
        "claims_html": claims_html,
        "rewrites_html": rewrites_html,
        "accept_all_btn": accept_all_btn,
        "clinician_md": clinician_md,
        "patient_md": patient_md,
        "edited_report": edited_report,
        "back_btn": back_btn,
        "_example_files": example_files,
    }


def _get_example_files() -> list[dict]:
    if not EXAMPLES_DIR.exists():
        return []
    examples = []
    manifest_path = EXAMPLES_DIR / "manifest.json"
    if manifest_path.exists():
        import json
        try:
            data = json.loads(manifest_path.read_text())
            return data.get("examples", [])
        except Exception:
            pass
    for folder in sorted(EXAMPLES_DIR.iterdir()):
        if folder.is_dir():
            img = next((f for f in folder.iterdir() if f.suffix.lower() in {".png",".jpg",".jpeg"}), None)
            rpt = next((f for f in folder.iterdir() if f.suffix.lower() in {".txt",".md"}), None)
            if img and rpt:
                examples.append({
                    "label": folder.name,
                    "image_path": str(img),
                    "report_path": str(rpt),
                })
    return examples
