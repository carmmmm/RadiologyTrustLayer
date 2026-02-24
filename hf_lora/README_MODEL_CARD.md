---
library_name: peft
base_model: google/medgemma-4b-it
tags:
  - medical
  - radiology
  - lora
  - peft
  - medgemma
  - rtl
  - healthcare
license: apache-2.0
---

# RTL LoRA — Radiology Trust Layer Adapter for MedGemma

This LoRA adapter extends `google/medgemma-4b-it` for **structured radiology report auditing**.

## Motivation

Base MedGemma produces excellent medical reasoning but sometimes:
1. Returns free-text instead of required JSON structure (schema compliance issues)
2. Uses overconfident language when uncertainty is warranted

This adapter addresses both issues through supervised fine-tuning on synthetic radiology QA pairs.

## Evaluation Results

| Metric | Base MedGemma | + RTL LoRA | Delta |
|--------|:------------:|:----------:|:-----:|
| JSON Schema Valid Rate | 84.0% | **100.0%** | +16.0% |
| Overconfidence Rate | 10.0% | **0.0%** | -10.0% |
| Label Value Valid Rate | 80.0% | **100.0%** | +20.0% |
| Label Accuracy | 65.3% | **87.3%** | +22.0% |
| Schema Repair Needed Rate | 84.0% | **0.0%** | -84.0% |

*Evaluated on 50 synthetic test cases.*

## Intended Use

Part of the [Radiology Trust Layer (RTL)](https://github.com/carmmmm/RadiologyTrustLayer) system for:
- Auditing radiology reports against imaging evidence
- Flagging overconfident or unsupported claims
- Generating calibrated rewrite suggestions

## How to Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

base_model_id = "google/medgemma-4b-it"
adapter_id = "outlawpink/rtl-medgemma-lora"

processor = AutoProcessor.from_pretrained(base_model_id, token="your_hf_token")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="your_hf_token"
)
model = PeftModel.from_pretrained(model, adapter_id)
model = model.merge_and_unload()  # Optional: merge for faster inference
```

## Training Details

- **Base model:** `google/medgemma-4b-it`
- **Method:** PEFT LoRA (r=8, alpha=16, target modules: q_proj, v_proj)
- **Quantization:** 8-bit (bitsandbytes)
- **Training data:** 200 synthetic radiology claim-alignment pairs (no PHI)
- **Evaluation data:** 50 synthetic test cases
- **Framework:** PEFT + TRL SFTTrainer
- **Hardware:** Kaggle T4 GPU (16GB VRAM)
- **Precision:** fp16

## Limitations

- Trained on synthetic data only — real-world performance may vary
- Not validated on real patient data
- Small evaluation set (50 cases) — results have wide confidence intervals
- Single-image analysis only (no multi-sequence or prior study comparison)
- **Not intended for clinical use**

## Disclaimer

This is a research artifact for the MedGemma Impact Challenge. Always consult
qualified radiologists for clinical decisions.
