"""History page — list and filter all prior runs for the current user."""
import gradio as gr


def build():
    gr.Markdown("## 📋 Audit History")

    with gr.Row():
        filter_severity = gr.Dropdown(
            choices=["All", "low", "medium", "high"],
            value="All",
            label="Filter by severity",
        )
        filter_score_min = gr.Slider(minimum=0, maximum=100, value=0, label="Min score", step=5)
        refresh_btn = gr.Button("Refresh", size="sm")

    history_table = gr.Dataframe(
        headers=["Date", "Case Label", "Score", "Severity", "Model", "Run ID"],
        datatype=["str", "str", "number", "str", "str", "str"],
        interactive=False,
        wrap=True,
    )

    gr.Markdown("*Click a row to open Audit Detail (paste the Run ID in the box below)*")
    with gr.Row():
        selected_run_id = gr.Textbox(label="Run ID to open", placeholder="Paste run_id here")
        open_detail_btn = gr.Button("Open Audit Detail")

    back_btn = gr.Button("Back to Home")

    return {
        "filter_severity": filter_severity,
        "filter_score_min": filter_score_min,
        "refresh_btn": refresh_btn,
        "history_table": history_table,
        "selected_run_id": selected_run_id,
        "open_detail_btn": open_detail_btn,
        "back_btn": back_btn,
    }
