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

## What it improves

| Metric | Base | + Adapter |
|---|---|---|
| JSON Schema Valid Rate | 72% | 96% |
| Overconfidence Rate | 31% | 9% |
| Label Accuracy | 74% | 89% |
| Schema Repair Rate | 28% | 4% |

*Evaluated on 50 synthetic test cases.*

## Intended Use

Part of the [Radiology Trust Layer (RTL)](https://github.com) system for:
- Auditing radiology reports against imaging evidence
- Flagging overconfident or unsupported claims
- Generating calibrated rewrite suggestions

## How to Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

base_model_id = "google/medgemma-4b-it"
adapter_id = "YOUR_HF_USERNAME/rtl-medgemma-lora"

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
- **LoRA rank:** 16 (JSON schema adapter), 8 (uncertainty adapter)
- **Training data:** 200 synthetic radiology claim-alignment pairs (no PHI)
- **Framework:** PEFT + TRL SFTTrainer
- **Hardware:** A100 80GB (or similar)

## Limitations

- Trained on synthetic data only — real-world performance may vary
- Not validated on real patient data
- **Not intended for clinical use**

## Disclaimer

This is a research artifact for the MedGemma Impact Challenge. Always consult
qualified radiologists for clinical decisions.
