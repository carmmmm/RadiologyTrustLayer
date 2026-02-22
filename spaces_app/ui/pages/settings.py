"""Settings page — user preferences and model configuration."""
import gradio as gr
from core import config


def build():
    gr.Markdown("## Settings")

    with gr.Accordion("Model Configuration", open=True):
        gr.Markdown(f"**Current base model:** `{config.MEDGEMMA_MODEL_ID}`")
        gr.Markdown(f"**Mock mode:** `{'enabled' if config.MEDGEMMA_MOCK else 'disabled'}`")
        gr.Markdown(f"**LoRA adapter:** `{config.RTL_LORA_ID or 'none'}`")
        gr.Markdown("*To change model settings, update environment variables and restart the Space.*")

    with gr.Accordion("Preferences", open=True):
        auto_patient = gr.Checkbox(label="Always generate patient explanation", value=True)
        show_confidence = gr.Checkbox(label="Show confidence scores in claim table", value=True)
        save_pref_btn = gr.Button("Save Preferences", size="sm")
        pref_status = gr.Markdown()

    with gr.Accordion("About RTL", open=False):
        gr.Markdown("""
**Radiology Trust Layer (RTL)** is a research demonstration built for the MedGemma Impact Challenge.

- Built with [MedGemma](https://huggingface.co/google/medgemma-4b-it) by Google Health AI
- Gradio app hosted on Hugging Face Spaces
- SQLite persistence for audit history
- Open-weight LoRA adapter for improved JSON compliance

**Disclaimer:** This is a research demonstration. It is NOT intended for clinical use.
Do not upload real patient data (PHI). Always consult qualified radiologists.
""")

    back_btn = gr.Button("Back to Home")

    return {
        "auto_patient": auto_patient,
        "show_confidence": show_confidence,
        "save_pref_btn": save_pref_btn,
        "pref_status": pref_status,
        "back_btn": back_btn,
    }
