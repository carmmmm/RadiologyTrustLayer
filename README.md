
# Radiology Trust Layer (RTL)

A MedGemma-powered multimodal auditing system that checks whether radiology reports are faithfully supported by imaging evidence. RTL extracts claims from free-text reports, analyzes the corresponding medical image with MedGemma's vision encoder, aligns each claim to visual findings, scores report accuracy, and generates actionable feedback for both clinicians and patients.

Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

## Links

| Resource | URL |
|----------|-----|
| Live Demo | [huggingface.co/spaces/outlawpink/RadiologyTrustLayer](https://huggingface.co/spaces/outlawpink/RadiologyTrustLayer) |
| LoRA Adapter Weights | [huggingface.co/outlawpink/rtl-medgemma-lora](https://huggingface.co/outlawpink/rtl-medgemma-lora) |
| Kaggle Notebook | [ [https://www.kaggle.com/code/olivecoco/radiology-trust-layer/edit](https://www.kaggle.com/code/olivecoco/radiology-trust-layer/edit) |
| Video Walkthrough | [PLACEHOLDER — update with YouTube URL] |
| Competition Page | [kaggle.com/competitions/medgemma-impact-challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) |
| Write-Up | [PLACEHOLDER — update with write-up URL] |
| GitHub Repository | [github.com/carmmmm/RadiologyTrustLayer](https://github.com/carmmmm/RadiologyTrustLayer) |

---

## How It Works

RTL runs a **6-step AI pipeline** on every radiology case:

| Step | Name | Input | Output |
|------|------|-------|--------|
| 1 | **Claim Extraction** | Report text | Structured list of clinical claims with character spans |
| 2 | **Image Findings** | Radiology image | Visual findings detected by MedGemma's vision encoder |
| 3 | **Alignment** | Claims + Findings | Each claim labeled: supported, uncertain, or needs_review |
| 4 | **Scoring** | Alignments | 0-100 safety score + severity classification (low/medium/high) |
| 5 | **Rewrite Suggestions** | Flagged claims + report | Calibrated rewrites for uncertain claims; verification flags for needs_review |
| 6 | **Summaries** | All above | Structured clinician summary + plain-language patient explanation |

Each step uses structured prompts with JSON schema validation to ensure consistent, parseable outputs.

---

## LoRA Training Results

The RTL LoRA adapter was fine-tuned on MedGemma-4B-IT using PEFT (r=4, target modules: q_proj, v_proj) with 8-bit quantization on a Kaggle T4 GPU. Evaluated on 50 synthetic radiology cases:

| Metric | Base MedGemma | + RTL LoRA | Delta |
|--------|:------------:|:----------:|:-----:|
| JSON Schema Valid Rate | 84.0% | **100.0%** | +16.0% |
| Overconfidence Rate | 10.0% | **0.0%** | -10.0% |
| Label Value Valid Rate | 80.0% | **100.0%** | +20.0% |
| Label Accuracy | 65.3% | **87.3%** | +22.0% |
| Schema Repair Needed Rate | 84.0% | **0.0%** | -84.0% |

Key improvements:
- **100% schema compliance** — the LoRA-adapted model always produces valid JSON matching the expected schema, eliminating the need for post-hoc repair
- **Zero overconfidence** — no instances of overconfident language in model outputs after fine-tuning
- **+22% label accuracy** — significantly better alignment between predicted labels and ground truth

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

- **`local`** -- Loads MedGemma-4B-IT directly via transformers (requires GPU)
- **`api`** -- Uses Hugging Face Inference API (no local GPU needed)
- **`mock`** -- Returns pre-built results for UI development and testing (set `MEDGEMMA_MOCK=true`)

---

## Project Structure

```
radiology-trust-layer/
  README.md
  requirements.txt
  .env.example                    # Environment variable template

  spaces_app/                     # Hugging Face Spaces entrypoint
    app.py                        # Gradio UI -- multi-page routing, CSS, event handlers
    ui/
      components.py               # HTML components: score gauge, flag counts, claim table
      render_report.py            # Sentence-level color-coded report highlighting
    data/examples/
      manifest.json               # 3 demo cases with metadata
      case_cxr_01/                # Right lower lobe pneumonia
      case_cxr_02/                # Congestive heart failure
      case_cxr_03/                # Normal study

  core/                           # Shared business logic
    config.py                     # Central config from environment variables
    pipeline/
      audit_pipeline.py           # 6-step orchestrator
      medgemma_client.py          # MedGemma inference with LoRA support
      prompts/v1/                 # 6 prompt templates
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
      parse_zip.py                # ZIP archive parsing (flat + folder layouts)
      runner.py                   # Sequential batch audit runner
    util/
      ids.py                      # UUID-based ID generation
      hashing.py                  # Image and text hashing for deduplication
      time.py                     # UTC timestamp helpers
      files.py                    # JSON/text file I/O
      validation.py               # JSON schema validation + text extraction

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
    smoke_test.py                 # End-to-end pipeline test

  kaggle/
    rtl_lora_training.ipynb       # Kaggle notebook for LoRA training
```

---

## Key Components

### Gradio UI (`spaces_app/app.py`)

Multi-page application with persistent navigation bar:

- **Landing** -- Overview of RTL with pipeline diagram and key metrics
- **Demo** -- Three pre-loaded chest X-ray cases with pre-computed results
- **Single Audit** -- Upload an image + report, run the full 6-step pipeline
- **Batch Audit** -- Upload a ZIP of cases for bulk processing
- **History** -- Browse and filter past audits (requires login)
- **Evaluation** -- Before/after LoRA metrics and model card
- **Settings** -- Runtime configuration and system information

The app is open-access (no login required to run audits). Login is optional for saving audit history.

### Audit Pipeline (`core/pipeline/`)

The pipeline orchestrator (`audit_pipeline.py`) runs 6 sequential inference calls through `medgemma_client.py`. Each step:

1. Loads a prompt template from `prompts/v1/`
2. Calls MedGemma with the prompt (+ image for step 2)
3. Validates the response against a JSON schema from `schemas/`
4. Attempts schema repair if validation fails (regex extraction, bracket fixing)
5. Returns structured data for the next step

### Scoring (`core/scoring/score.py`)

Computes a 0-100 safety score from alignment labels:
- **Supported** claims: no penalty
- **Uncertain** claims: partial penalty (8 points)
- **Needs review** claims: full penalty (25 points)

Severity levels: **low** (>= 80), **medium** (>= 50), **high** (< 50).

### LoRA Fine-Tuning (`hf_lora/`)

Two adapter objectives:

1. **JSON Schema Compliance** -- Improves structured output generation (valid JSON, correct schema adherence)
2. **Uncertainty Calibration** -- Reduces overconfident language in favor of calibrated hedging

Training uses PEFT LoRA adapters on MedGemma-4B-IT with TRL's SFTTrainer and 8-bit quantization. Synthetic training data is generated from template clinical scenarios.

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
| `HF_TOKEN` | -- | Required for gated model access |
| `RTL_LORA_ID` | -- | HF repo ID of LoRA adapter (optional) |
| `RTL_PROMPT_VERSION` | `v1` | Prompt template version |

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
|------|-----------|:-------------:|:-----------------:|
| CXR 01 | Right lower lobe pneumonia | 75-90 | Low |
| CXR 02 | Congestive heart failure | 55-75 | Medium |
| CXR 03 | Normal study | 88-100 | Low |

---

## Disclaimer

RTL is a **research demonstration** for the MedGemma Impact Challenge. It is NOT intended for clinical use. Do not upload real patient data. Always consult qualified radiologists for medical decisions.

---

## License

Apache 2.0
