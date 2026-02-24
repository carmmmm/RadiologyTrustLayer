"""Push trained LoRA adapter to Hugging Face Hub."""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def push_adapter(
    adapter_path: str,
    repo_id: str,
    base_model_id: str = "google/medgemma-4b-it",
    token: str = "",
    private: bool = False,
) -> str:
    """Upload adapter weights and model card to HF Hub."""
    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token or None)

    # Create repo if needed
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
        logger.info("Repo created/verified: %s", repo_id)
    except Exception as e:
        logger.warning("Could not create repo: %s", e)

    # Write model card
    card_path = Path(adapter_path) / "README.md"
    if not card_path.exists():
        card_content = _generate_model_card(repo_id, base_model_id, adapter_path)
        card_path.write_text(card_content)

    url = upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        token=token or None,
        commit_message="Upload RTL LoRA adapter",
    )
    logger.info("Uploaded to: %s", url)
    return url


def _generate_model_card(repo_id: str, base_model: str, adapter_path: str) -> str:
    summary_path = Path(adapter_path) / "training_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return f"""---
library_name: peft
base_model: {base_model}
tags:
  - medical
  - radiology
  - lora
  - peft
  - medgemma
  - rtl
license: apache-2.0
---

# RTL LoRA Adapter for MedGemma

This is a LoRA adapter for `{base_model}` trained for the **Radiology Trust Layer (RTL)** project.

## What this adapter does

- Improves **JSON schema compliance** in structured output tasks (84% -> 100%)
- Reduces **overconfident language** in uncertainty alignment tasks (10% -> 0%)
- Improves **label accuracy** for claim-evidence alignment (65.3% -> 87.3%)
- Trained on synthetic radiology QA pairs (no PHI)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
model = model.merge_and_unload()
```

## Training Details

- Base model: `{base_model}`
- Method: PEFT LoRA (r=8, alpha=16, target modules: q_proj, v_proj)
- Training samples: {summary.get('train_samples', 'N/A')}
- Framework: PEFT + TRL SFTTrainer
- Hardware: Kaggle T4 GPU

## Evaluation

| Metric | Base | + LoRA | Delta |
|--------|:----:|:------:|:-----:|
| JSON Schema Valid Rate | 84.0% | **100.0%** | +16.0% |
| Overconfidence Rate | 10.0% | **0.0%** | -10.0% |
| Label Value Valid Rate | 80.0% | **100.0%** | +20.0% |
| Label Accuracy | 65.3% | **87.3%** | +22.0% |
| Schema Repair Needed Rate | 84.0% | **0.0%** | -84.0% |

See the [RTL GitHub repository](https://github.com/carmmmm/RadiologyTrustLayer) for full evaluation details.

## Disclaimer

This adapter is for **research purposes only**. Not intended for clinical use.
"""


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Local path to adapter checkpoint")
    parser.add_argument("--repo-id", required=True, help="HF Hub repo id (e.g. yourname/rtl-lora)")
    parser.add_argument("--base-model", default="google/medgemma-4b-it")
    parser.add_argument("--token", default="", help="HF API token")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    url = push_adapter(
        adapter_path=args.adapter_path,
        repo_id=args.repo_id,
        base_model_id=args.base_model,
        token=args.token,
        private=args.private,
    )
    print(f"Adapter published: {url}")


if __name__ == "__main__":
    main()
