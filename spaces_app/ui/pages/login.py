"""Login / Create Account page."""
import gradio as gr


def build():
    gr.Markdown("# Radiology Trust Layer (RTL)")
    gr.Markdown(
        "**MedGemma-powered radiology report auditing.** "
        "Log in or create a demo account to get started.\n\n"
        "🔒 **No PHI** — do not upload real patient data."
    )

    with gr.Tab("Log In"):
        login_email = gr.Textbox(label="Email", placeholder="name@example.com")
        login_pw = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Log In", variant="primary")
        login_msg = gr.Markdown()

    with gr.Tab("Create Account"):
        create_email = gr.Textbox(label="Email")
        create_name = gr.Textbox(label="Display name", placeholder="Dr. Smith")
        create_pw = gr.Textbox(label="Password", type="password")
        create_btn = gr.Button("Create Account", variant="primary")
        create_msg = gr.Markdown()

    return {
        "login_email": login_email,
        "login_pw": login_pw,
        "login_btn": login_btn,
        "login_msg": login_msg,
        "create_email": create_email,
        "create_name": create_name,
        "create_pw": create_pw,
        "create_btn": create_btn,
        "create_msg": create_msg,
    }
