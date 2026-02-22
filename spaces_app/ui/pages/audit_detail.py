"""Audit Detail page — display a saved run from the database."""
import gradio as gr

from spaces_app.ui.components import score_gauge_html, flag_counts_html, claim_table_html, rewrite_suggestions_html
from spaces_app.ui.render_report import render_highlighted_report


def build():
    gr.Markdown("## 🔍 Audit Detail")

    run_id_display = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Score")
            score_html = gr.HTML()
            flag_html = gr.HTML()
            gr.Markdown("### Case Info")
            meta_md = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Report Highlights"):
                    report_html = gr.HTML()
                with gr.Tab("Claim Analysis"):
                    claims_html = gr.HTML()
                with gr.Tab("Suggested Rewrites"):
                    rewrites_html = gr.HTML()
                with gr.Tab("Clinician Summary"):
                    clinician_md = gr.Markdown()
                with gr.Tab("Patient Explanation"):
                    patient_md = gr.Markdown()
                with gr.Tab("Edited Report"):
                    edited_report_txt = gr.Textbox(label="Edited Report", lines=10, interactive=False)
                with gr.Tab("Audit Trail"):
                    audit_trail_html = gr.HTML()

    with gr.Row():
        export_btn = gr.Button("Export JSON")
        export_file = gr.File(label="Download", visible=False)

    back_btn = gr.Button("Back to Home")

    return {
        "run_id_display": run_id_display,
        "score_html": score_html,
        "flag_html": flag_html,
        "meta_md": meta_md,
        "report_html": report_html,
        "claims_html": claims_html,
        "rewrites_html": rewrites_html,
        "clinician_md": clinician_md,
        "patient_md": patient_md,
        "edited_report_txt": edited_report_txt,
        "audit_trail_html": audit_trail_html,
        "export_btn": export_btn,
        "export_file": export_file,
        "back_btn": back_btn,
    }
