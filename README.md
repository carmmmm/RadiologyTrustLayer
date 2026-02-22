---
title: Radiology Trust Layer
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.31.0"
python_version: "3.11"
app_file: spaces_app/app.py
pinned: false
---

# Radiology Trust Layer (RTL)

A MedGemma-powered multimodal auditing system that checks whether radiology reports are faithfully supported by imaging evidence. RTL extracts claims from free-text reports, analyzes the corresponding medical image, aligns each claim to visual findings, scores report accuracy, and generates actionable feedback for clinicians and patients.

Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

**Live Demo:** [huggingface.co/spaces/outlawpink/RadiologyTrustLayer](https://huggingface.co/spaces/outlawpink/RadiologyTrustLayer)

**Video Tutorial:**


**Kaggle Notebook:**


**Hugging Face Model:**
---

## How It Works

RTL runs a **6-step AI pipeline** on every radiology case:

| Step | Name | Input | Output |
|------|------|-------|--------|
| 1 | **Claim Extraction** | Report text | Structured list of clinical claims |
| 2 | **Image Findings** | Radiology image | Visual findings detected by MedGemma |
| 3 | **Alignment** | Claims + Findings | Each claim labeled: supported, uncertain, needs_review |
| 4 | **Scoring** | Alignments | 0-100 safety score + severity (low/medium/high) |
| 5 | **Rewrite Suggestions** | Flagged claims + report | Suggested corrections for uncertain/flagged claims |
| 6 | **Clinician Summary + Patient Explanation** | All above | Actionable summary for clinicians; plain-language version for patients |

Each step uses structured prompts with JSON schema validation to ensure consistent, parseable outputs.

---

## Architecture

```
spaces_app/app.py          Gradio UI (multi-page, Google-style design)
    |
core/pipeline/
    audit_pipeline.py       6-step orchestrator with progress callbacks
    medgemma_client.py      MedGemma inference (local / API / mock modes)
    prompts/v1/*.md         Prompt templates for each pipeline step
    schemas/*.schema.json   JSON schemas for structured output validation
    |
core/scoring/score.py      0-100 scoring with severity classification
core/db/                   SQLite persistence (users, runs, audit trail)
core/batch/                ZIP-based batch processing
    |
hf_lora/                   LoRA fine-tuning pipeline (PEFT + TRL)
```

### Inference Modes

RTL supports three inference modes, set via `MEDGEMMA_INFERENCE_MODE`:

- **`local`** — Loads MedGemma-4B-IT directly via transformers (requires GPU)
- **`api`** — Uses Hugging Face Inference API (no local GPU needed)
- **`mock`** — Returns pre-built results for UI development and testing (set `MEDGEMMA_MOCK=true`)

---

## Project Structure

```
radiology-trust-layer/
  README.md
  requirements.txt
  .env.example                    # Environment variable template

  spaces_app/                     # Hugging Face Spaces entrypoint
    app.py                        # Gradio UI — multi-page routing, CSS, event handlers
    ui/
      components.py               # HTML components: score gauge, flag counts, claim table
      render_report.py            # Sentence-level color-coded report highlighting
    data/examples/
      manifest.json               # 3 demo cases with metadata
      case_cxr_01/                # Right lower lobe pneumonia (real X-ray)
      case_cxr_02/                # Congestive heart failure (real X-ray)
      case_cxr_03/                # Normal study (real X-ray)
    storage/                      # Runtime SQLite + outputs (not committed)

  core/                           # Shared business logic
    config.py                     # Central config from environment variables
    pipeline/
      audit_pipeline.py           # 6-step orchestrator
      medgemma_client.py          # MedGemma inference with LoRA support
      prompts/v1/                 # 6 prompt templates (claim_extraction, image_findings, etc.)
      schemas/                    # 6 JSON schemas for structured outputs
    scoring/
      score.py                    # 0-100 score computation + severity classification
    db/
      schema.sql                  # SQLite schema (users, runs, batches, events)
      db.py                       # Thread-safe SQLite connection management
      repo.py                     # CRUD operations for all tables
    audit_trail/
      events.py                   # Typed audit event logging
    batch/
      parse_zip.py                # ZIP archive parsing
      runner.py                   # Sequential batch audit runner
    util/
      ids.py                      # UUID-based run ID generation
      hashing.py                  # Image and text hashing for deduplication
      time.py                     # UTC timestamp helpers
      files.py                    # JSON/text file I/O
      validation.py               # JSON schema validation with repair

  hf_lora/                        # LoRA fine-tuning pipeline
    train_lora.py                 # PEFT + TRL SFTTrainer training script
    dataset/
      make_synthetic.py           # Generate synthetic training pairs
      format.py                   # Convert to Gemma chat template format
    configs/
      lora_json.schema.yaml       # Config: JSON schema compliance adapter
      lora_uncertainty.yaml       # Config: Uncertainty calibration adapter
    eval/
      eval_lora_before_after.py   # Before/after evaluation metrics
    publish/
      push_to_hf.py              # Upload adapter + model card to HF Hub
    README_MODEL_CARD.md          # Model card template

  eval/                           # Evaluation evidence
    rubric.md                     # Competition rubric mapping
    sample_outputs/
      mock_results.json           # Complete mock audit result

  scripts/
    init_space_storage.py         # Creates SQLite DB + storage dirs at runtime
    smoke_test.py                 # End-to-end pipeline test (scoring, validation, DB, audit)

  video/
    script.md                     # 3-minute demo video script
    storyboard.md                 # Shot-by-shot storyboard
```

---

## Key Components

### Gradio UI (`spaces_app/app.py`)

Multi-page application with persistent navigation bar. Pages:

- **Single Audit** — Upload an image + report, run the full 6-step pipeline, view results across tabs (report highlights, claim analysis, rewrites, clinician summary, patient explanation)
- **Batch Audit** — Upload a ZIP of cases for bulk processing
- **History** — Browse and filter past audits (requires login)
- **Evaluation** — Model card, before/after LoRA metrics, example cases
- **Settings** — Current configuration display

The app is open-access (no login required to run audits). Login is optional for saving audit history. Three pre-loaded example cases let new users try the system immediately.

### Audit Pipeline (`core/pipeline/`)

The pipeline orchestrator (`audit_pipeline.py`) runs 6 sequential inference calls through `medgemma_client.py`. Each step:

1. Loads a prompt template from `prompts/v1/`
2. Calls MedGemma with the prompt (+ image for step 2)
3. Validates the response against a JSON schema from `schemas/`
4. Attempts schema repair if validation fails (regex extraction, bracket fixing)
5. Returns structured data for the next step

### Scoring (`core/scoring/score.py`)

Computes a 0-100 safety score from alignment labels:
- **Supported** claims contribute positively
- **Uncertain** claims receive partial deduction
- **Needs review** claims receive full deduction
- **Not assessable** claims are noted but not penalized

Severity levels: **low** (score >= 80), **medium** (score >= 50), **high** (score < 50).

### LoRA Fine-Tuning (`hf_lora/`)

Two adapter objectives:

1. **JSON Schema Compliance** — Improves structured output generation (valid JSON, correct schema adherence)
2. **Uncertainty Calibration** — Reduces overconfident language ("definitely", "clearly") in favor of calibrated hedging

Training uses PEFT LoRA adapters on MedGemma-4B-IT with TRL's SFTTrainer. Synthetic training data is generated from template clinical scenarios.

---

## Quick Start

### Local Development (Mock Mode)

```bash
# Clone and install
git clone https://github.com/carmmmm/RadiologyTrustLayer.git
cd RadiologyTrustLayer
pip install -r requirements.txt

# Run smoke test
MEDGEMMA_MOCK=true python scripts/smoke_test.py

# Launch the UI
MEDGEMMA_MOCK=true python spaces_app/app.py
# Open http://127.0.0.1:7860
```

### With Real MedGemma (GPU Required)

```bash
# Set your HF token (MedGemma is a gated model)
export HF_TOKEN=your_token_here
export MEDGEMMA_MOCK=false
export MEDGEMMA_INFERENCE_MODE=local   # or "api" for HF Inference API

python spaces_app/app.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDGEMMA_MOCK` | `false` | Skip model loading, use pre-built results |
| `MEDGEMMA_INFERENCE_MODE` | `local` | `local`, `api`, or set MOCK=true |
| `MEDGEMMA_MODEL_ID` | `google/medgemma-4b-it` | Base model ID |
| `HF_TOKEN` | — | Required for gated model access |
| `RTL_LORA_ID` | — | HF repo ID of LoRA adapter (optional) |
| `RTL_PROMPT_VERSION` | `v1` | Prompt template version |

---

## LoRA Training on Kaggle

Training runs on Kaggle's free T4 GPUs. See the Kaggle notebook for the complete workflow:

1. Generate synthetic training data
2. Format for Gemma chat template
3. Fine-tune with 8-bit quantization + LoRA (r=4, target: q_proj, v_proj)
4. Evaluate before/after metrics
5. Publish adapter to Hugging Face Hub

```bash
# Generate data
python hf_lora/dataset/make_synthetic.py

# Train
python hf_lora/train_lora.py --config hf_lora/configs/lora_json.schema.yaml

# Evaluate
python hf_lora/eval/eval_lora_before_after.py

# Publish
python hf_lora/publish/push_to_hf.py \
  --adapter-path hf_lora/checkpoints/json_schema \
  --repo-id outlawpink/rtl-medgemma-lora
```

---

## Deployment

### Hugging Face Spaces

The app is deployed at [outlawpink/RadiologyTrustLayer](https://huggingface.co/spaces/outlawpink/RadiologyTrustLayer).

Required Space settings:
- **Hardware**: Nvidia T4 small (16GB VRAM)
- **Secrets**: `HF_TOKEN`
- **Variables**: `MEDGEMMA_MOCK=false`, `MEDGEMMA_INFERENCE_MODE=local`

---

## Demo Cases

Three pre-loaded chest X-ray cases (real images from the [chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) dataset):

| Case | Condition | Expected Score | Expected Severity |
|------|-----------|---------------|-------------------|
| CXR 01 | Right lower lobe pneumonia | 75-90 | Low |
| CXR 02 | Congestive heart failure | 55-75 | Medium |
| CXR 03 | Normal study | 88-100 | Low |

---

## Disclaimer

RTL is a **research demonstration** for the MedGemma Impact Challenge. It is NOT intended for clinical use. Do not upload real patient data. Always consult qualified radiologists for medical decisions.

---

## License

Apache 2.0
