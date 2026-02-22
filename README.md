---
title: Radiology Trust Layer
emoji: "🏥"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.6.0"
app_file: spaces_app/app.py
pinned: true
license: apache-2.0
---

# Radiology Trust Layer (RTL)

MedGemma-powered radiology report auditing. Checks whether radiology reports are faithfully supported by imaging evidence using a 6-step AI pipeline.

Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

**Folder Structure:**

radiology-trust-layer/
  .gitignore
  README.md
  LICENSE
  requirements.txt
  pyproject.toml
  .env.example

  spaces_app/                       # Hugging Face Spaces entrypoint
    app.py                          # Gradio UI + routing
    ui/
      components.py                 # reusable UI blocks (tables, badges, dialogs)
      render_report.py              # sentence highlighting renderer
      pages/
        login.py                    # login/create account UI
        home.py                     # HOL-simple tiles + recent history
        single_audit.py             # upload image+report -> run audit
        batch_audit.py              # upload ZIP -> run batch -> results table
        audit_detail.py             # image left, report right, accept rewrites, tabs
        history.py                  # filters + open prior runs
        evaluation.py               # show metrics + evidence
        settings.py                 # persist inputs toggle etc.
    assets/
      logo.svg
      screenshots/                  # for the Space README panel
    data/
      examples/
        manifest.json
        case_cxr_01/
          image.png
          report.txt
        case_cxr_02/
          image.png
          report.txt
        case_cxr_03/
          image.png
          report.txt
    storage/
      .gitkeep                      # runtime-created sqlite + outputs (not committed)

  core/                             # shared business logic used by Space
    config.py
    util/
      ids.py
      hashing.py
      time.py
      files.py
      validation.py
    db/
      schema.sql
      db.py
      repo.py                       # users/runs/batches/events CRUD
    scoring/
      score.py                      # 0–100 + severity
    audit_trail/
      events.py                     # event types + log helper
    batch/
      parse_zip.py                  # parse zip structure
      runner.py                     # sequential/concurrent batch runner
    pipeline/
      audit_pipeline.py             # orchestrator (calls medgemma client)
      medgemma_client.py            # MedGemma inference + optional LoRA
      prompts/
        v1/
          claim_extraction.md
          image_findings.md
          alignment.md
          rewrite.md
          clinician_summary.md
          patient_explain.md
      schemas/
        claim_extraction.schema.json
        image_findings.schema.json
        alignment.schema.json
        rewrite.schema.json
        clinician_summary.schema.json
        patient_explain.schema.json

  hf_lora/                          # Bonus: open-weight tracing artifact
    README_MODEL_CARD.md            # model card template (will be copied to HF repo)
    train_lora.py                   # LoRA training script
    dataset/
      make_synthetic.py             # build synthetic pairs from public examples
      format.py                     # convert to training JSONL
    configs/
      lora_uncertainty.yaml
      lora_json_schema.yaml
    eval/
      eval_lora_before_after.py     # schema validity + overconfidence reduction
    publish/
      push_to_hf.py                 # helper to upload adapter weights + card

  eval/                             # evaluation evidence for judges not fully layed out, template used for all Kaggle.
    rubric.md
    results/
      summary.md
      tables/
        metrics.csv
      figures/
        success_case_1.png
        failure_case_1.png
    scripts/
      run_eval.py                   # runs on example set + logs results
    sample_outputs/
      mock_results.json             # use to build UI before models work

  video/
    script.md                       # exact 3-min script + shot list
    storyboard.md                   # tight plan to hit rubric points

  scripts/
    init_space_storage.py           # creates sqlite + folders at runtime
    smoke_test.py                   # runs core pipeline on examples (CI-friendly)


**HF Spaces are public. So:**

This is not HIPAA.

You enforce: NO PHI + demo inputs only if you want.

Accounts exist to show “prior runs” and workflow credibility, not actual hospital deployme



**Claude Update 2.18.16**

What was implemented (3,554 lines across 47 Python files + 10 JSON + 11 Markdown)
Core Pipeline
core/config.py — Central config from env vars
core/util/ — IDs, hashing, timestamps, file I/O, JSON validation with retry
core/db/db.py + repo.py — Thread-safe SQLite with full CRUD
core/audit_trail/events.py — Typed event constants + logging
core/scoring/score.py — 0–100 score + severity (low/medium/high)
core/pipeline/medgemma_client.py — MedGemma inference with 3 modes: local (transformers), API (HF Inference), mock (for dev)
core/pipeline/audit_pipeline.py — 6-step orchestrator with progress callbacks
6 prompt templates + 6 JSON schemas for all pipeline steps
core/batch/ — ZIP parsing + sequential batch runner
Gradio UI
spaces_app/app.py — Fully rewired multi-page Gradio app (fixed the broken gr.Group() output bug from original)
All 8 pages: Login, Home, Single Audit, Batch Audit, Audit Detail, History, Evaluation, Settings
Highlighted report rendering with color-coded claim labels
Reusable components: score gauge, flag counts, claim table, rewrite suggestions
LoRA Training Pipeline
hf_lora/train_lora.py — Full PEFT+TRL training script
Synthetic dataset generation + chat-template formatting
Before/after evaluation with JSON validity, overconfidence, label accuracy metrics
push_to_hf.py + model card for HF Hub publication
Demo Assets
3 real chest X-ray images downloaded from HuggingFace (pneumonia × 2, normal × 1)
Corresponding synthetic reports for each case
eval/sample_outputs/mock_results.json — Complete mock run for UI testing
Smoke test: all 4 test suites passing
Next steps to deploy
Set HF_TOKEN env var in HF Spaces secrets
Set MEDGEMMA_MOCK=false and MEDGEMMA_INFERENCE_MODE=local (needs GPU Space)
Run python hf_lora/dataset/make_synthetic.py then python hf_lora/train_lora.py --config hf_lora/configs/lora_json.schema.yaml
Publish LoRA: python hf_lora/publish/push_to_hf.py --adapter-path hf_lora/checkpoints/json_schema --repo-id YOUR_HF_USERNAME/rtl-medgemma-lora
Record the 3-minute demo video using video/script.md