---
library_name: peft
base_model: google/medgemma-4b-it
pipeline_tag: text-generation
tags:
  - medical
  - radiology
  - lora
  - peft
  - medgemma
  - rtl
  - healthcare
  - structured-output
  - uncertainty-calibration
  - medgemma-impact-challenge
license: apache-2.0
datasets:
  - synthetic
model-index:
  - name: rtl-medgemma-lora
    results:
      - task:
          type: structured-output-generation
          name: JSON Schema Compliance
        metrics:
          - type: accuracy
            name: JSON Schema Valid Rate
            value: 100.0
          - type: accuracy
            name: Label Accuracy
            value: 87.3
---

# RTL LoRA -- Radiology Trust Layer Adapter for MedGemma

A PEFT LoRA adapter that extends [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) for **structured radiology report auditing**. Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) on Kaggle.

## Model Description

The **Radiology Trust Layer (RTL)** is a 6-step AI pipeline that audits radiology reports against imaging evidence. It extracts claims from free-text reports, analyzes the corresponding medical image with MedGemma's vision encoder, and aligns each claim to visual findings.

This adapter addresses two reliability gaps in the base MedGemma model when used for structured auditing:

1. **Schema compliance** -- Base MedGemma sometimes returns free-text narrative instead of the required JSON structure, breaking downstream pipeline steps. The adapter achieves 100% valid JSON output.
2. **Uncertainty calibration** -- Base MedGemma occasionally uses overconfident language ("definitely," "clearly") when evidence is ambiguous. The adapter eliminates this, producing calibrated hedging appropriate for clinical contexts.

## Evaluation Results

Evaluated on 50 held-out synthetic test cases across two task types (JSON schema compliance and uncertainty calibration):

| Metric | Base MedGemma | + RTL LoRA | Delta |
|--------|:------------:|:----------:|:-----:|
| JSON Schema Valid Rate | 84.0% | **100.0%** | +16.0% |
| Overconfidence Rate | 10.0% | **0.0%** | -10.0% |
| Label Value Valid Rate | 80.0% | **100.0%** | +20.0% |
| Label Accuracy | 65.3% | **87.3%** | +22.0% |
| Schema Repair Needed Rate | 84.0% | **0.0%** | -84.0% |

### Metric Definitions

- **JSON Schema Valid Rate**: Percentage of model outputs that parse as valid JSON with the expected `alignments` array structure.
- **Overconfidence Rate**: Percentage of outputs containing overconfident language patterns (e.g., "definitely," "clearly," "no doubt") that are inappropriate for uncertain clinical findings.
- **Label Value Valid Rate**: Percentage of predicted alignment labels that fall within the valid label set (`supported`, `uncertain`, `needs_review`).
- **Label Accuracy**: Agreement rate between predicted alignment labels and ground-truth labels, computed per-claim.
- **Schema Repair Needed Rate**: Percentage of outputs that required regex-based extraction to recover valid JSON (indicating the model wrapped JSON in markdown or narrative text).

## How to Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

base_model_id = "google/medgemma-4b-it"
adapter_id = "outlawpink/rtl-medgemma-lora"

# Load base model (requires HF token for gated model access)
processor = AutoProcessor.from_pretrained(base_model_id, token="YOUR_HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="YOUR_HF_TOKEN",
)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(model, adapter_id)
model = model.merge_and_unload()  # Merge for faster inference

# Inference
prompt = """Align the following claims to the image findings.
Claims: [{"claim_id": "c1", "text": "There is consolidation in the right lower lobe."}]
Respond with JSON: {"alignments": [{"claim_id": "...", "label": "supported|uncertain|needs_review", "evidence": "...", "confidence": 0.0-1.0}]}"""

inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

## Training Details

### Training Data

The adapter was trained on **synthetic** radiology claim-alignment pairs. No real patient data or protected health information (PHI) was used at any stage.

- **Training set**: 800 examples (200 base pairs x 4 augmentations)
- **Evaluation set**: 100 examples (50 per task)
- **Task 1 -- JSON schema compliance**: Pairs of (radiology alignment prompt, correctly structured JSON response). Claims are generated from templates covering 10 finding types, 7 anatomical locations, and 7 diagnoses.
- **Task 2 -- Uncertainty calibration**: Pairs of (overconfident phrasing, calibrated phrasing) for common radiology language patterns.

### Training Procedure

| Parameter | Value |
|-----------|-------|
| Base model | `google/medgemma-4b-it` (4B parameters) |
| Method | PEFT LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Quantization | 8-bit (bitsandbytes) |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 2e-4 |
| Scheduler | Cosine with warmup |
| Precision | fp16 |
| Framework | PEFT 0.18.1 + TRL SFTTrainer |
| Hardware | Kaggle T4 GPU (16GB VRAM) |
| Training time | ~15 minutes |
| Trainable parameters | ~0.08% of base model |

### Chat Template

Training data is formatted using the Gemma chat template:

```
<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
{completion}<end_of_turn>
```

## Intended Use

This adapter is designed for use within the [Radiology Trust Layer](https://huggingface.co/spaces/outlawpink/RadiologyTrustLayer) system, specifically for:

- Auditing radiology reports against imaging evidence
- Producing structured JSON outputs for pipeline integration
- Generating calibrated alignment labels (`supported`, `uncertain`, `needs_review`)
- Reducing overconfident language in clinical text analysis

### Out-of-Scope Use

- **Clinical decision-making**: This adapter has not been validated on real patient data and must not be used for diagnostic purposes.
- **General medical QA**: The adapter is specialized for the RTL structured output format and may not improve general medical question answering.
- **Non-radiology domains**: Training data covers chest X-ray findings only.

## Limitations

- **Synthetic training data**: All training examples are template-generated. Real-world radiology reports exhibit greater linguistic diversity, abbreviations, and domain-specific conventions not captured in synthetic data.
- **Small evaluation set**: 50 test cases per task. Results have wide confidence intervals and may not generalize to all radiology scenarios.
- **Single modality**: Evaluated on text-only alignment tasks. The full RTL pipeline includes image analysis (Step 2), but LoRA training targeted text-based steps only.
- **Chest X-ray focus**: Training templates cover chest radiography findings. Performance on other imaging modalities (CT, MRI, ultrasound) is untested.
- **English only**: All training and evaluation data is in English.

## Related Resources

| Resource | Link |
|----------|------|
| Live Demo | [RTL on Hugging Face Spaces](https://huggingface.co/spaces/outlawpink/RadiologyTrustLayer) |
| Source Code | [GitHub Repository](https://github.com/carmmmm/RadiologyTrustLayer) |
| Training Notebook | [Kaggle Notebook](https://www.kaggle.com/code/olivecoco/radiology-trust-layer) |
| Competition | [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) |
| Base Model | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |

## Citation

```bibtex
@misc{rtl-lora-2025,
  title={RTL LoRA: Radiology Trust Layer Adapter for MedGemma},
  author={Carmen},
  year={2025},
  url={https://huggingface.co/outlawpink/rtl-medgemma-lora},
  note={MedGemma Impact Challenge submission}
}
```

## Disclaimer

This adapter is a **research demonstration** for the MedGemma Impact Challenge. It is **not intended for clinical use**. Do not use it to make medical decisions. Always consult qualified radiologists for clinical interpretations.
