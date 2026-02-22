"""Batch Audit page — upload ZIP of cases, run pipeline, show table."""
import gradio as gr


def build():
    gr.Markdown("## 📦 Batch Audit")
    gr.Markdown(
        "Upload a ZIP archive containing multiple radiology cases. "
        "Each case needs an image file and a `.txt` report. "
        "See the [README](https://github.com) for expected structure."
    )

    with gr.Row():
        with gr.Column(scale=1):
            zip_input = gr.File(
                label="Upload ZIP archive",
                file_types=[".zip"],
            )
            batch_label = gr.Textbox(label="Batch label (optional)", placeholder="e.g. ICU_Q1_2025")
            use_lora_batch = gr.Checkbox(label="Use RTL LoRA adapter", value=False)
            run_batch_btn = gr.Button("Run Batch Audit", variant="primary", size="lg")
            batch_status = gr.Markdown()

        with gr.Column(scale=2):
            gr.Markdown("### Progress")
            batch_progress = gr.Progress()
            progress_log = gr.Markdown()

    gr.Markdown("---\n### Batch Summary")
    with gr.Row():
        avg_score_md = gr.Markdown()
        pct_review_md = gr.Markdown()
        severity_dist_md = gr.Markdown()

    gr.Markdown("### Per-Case Results")
    batch_table = gr.Dataframe(
        headers=["Case ID", "Score", "Severity", "Supported", "Uncertain", "Needs Review", "Run ID"],
        datatype=["str", "number", "str", "number", "number", "number", "str"],
        interactive=False,
        wrap=True,
    )

    back_btn = gr.Button("Back to Home")

    return {
        "zip_input": zip_input,
        "batch_label": batch_label,
        "use_lora_batch": use_lora_batch,
        "run_batch_btn": run_batch_btn,
        "batch_status": batch_status,
        "batch_progress": batch_progress,
        "progress_log": progress_log,
        "avg_score_md": avg_score_md,
        "pct_review_md": pct_review_md,
        "severity_dist_md": severity_dist_md,
        "batch_table": batch_table,
        "back_btn": back_btn,
    }
