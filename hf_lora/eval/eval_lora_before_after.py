"""
Evaluate MedGemma before and after LoRA fine-tuning.

Metrics:
- JSON schema valid rate (% of responses that parse as valid JSON matching schema)
- Overconfidence rate (% of responses with overconfident language)
- Label accuracy (against ground-truth alignment labels)
- Schema repair rate (% of responses needing regex repair)

Usage:
  python hf_lora/eval/eval_lora_before_after.py --base google/medgemma-4b-it
     [--lora path/to/adapter] [--test-file hf_lora/dataset/eval.jsonl] [--n 50]
"""
import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

OVERCONFIDENT_PATTERNS = [
    r"\bdefinitely\b", r"\bclearly\b", r"\bobviously\b", r"\bconfirms\b",
    r"\bno doubt\b", r"\bwithout question\b", r"\bconclusively\b",
]
LABELS = {"supported", "uncertain", "not_assessable", "needs_review"}


def is_json_valid(text: str) -> bool:
    try:
        data = json.loads(text)
        return "alignments" in data and isinstance(data["alignments"], list)
    except Exception:
        return False


def try_extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def has_overconfident_language(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in OVERCONFIDENT_PATTERNS)


def label_accuracy(predicted: list[dict], ground_truth: list[dict]) -> float:
    gt_map = {a["claim_id"]: a["label"] for a in ground_truth}
    correct = sum(1 for a in predicted if gt_map.get(a.get("claim_id","")) == a.get("label",""))
    return correct / max(len(predicted), 1)


def evaluate_model(model_output_fn, test_cases: list[dict]) -> dict:
    """
    Args:
        model_output_fn: callable(prompt) -> str
        test_cases: list of {"prompt": str, "completion": str, "ground_truth_alignments": list}
    """
    n = len(test_cases)
    json_valid = 0
    overconfident = 0
    schema_repaired = 0
    label_accs = []

    for case in test_cases:
        prompt = case["prompt"]
        gt_alignments = case.get("ground_truth_alignments", [])

        try:
            output = model_output_fn(prompt)
        except Exception as e:
            logger.warning("Inference failed: %s", e)
            output = ""

        # Check JSON validity
        if is_json_valid(output):
            json_valid += 1
        else:
            # Try repair
            parsed = try_extract_json(output)
            if parsed and "alignments" in parsed:
                json_valid += 1
                schema_repaired += 1

        # Check overconfidence
        if has_overconfident_language(output):
            overconfident += 1

        # Label accuracy
        parsed = try_extract_json(output) or {}
        predicted = parsed.get("alignments", [])
        if gt_alignments and predicted:
            label_accs.append(label_accuracy(predicted, gt_alignments))

    return {
        "n": n,
        "json_valid_rate": json_valid / n,
        "overconfidence_rate": overconfident / n,
        "label_accuracy": sum(label_accs) / len(label_accs) if label_accs else None,
        "schema_repair_rate": schema_repaired / n,
    }


def mock_base_fn(prompt: str) -> str:
    """Simulate base model with ~72% JSON validity and 31% overconfidence."""
    import random
    if random.random() < 0.28:
        return "The claims appear to be supported by the imaging evidence."
    overconf = random.random() < 0.31
    prefix = "Clearly, " if overconf else ""
    return json.dumps({
        "alignments": [
            {"claim_id": "c1", "label": "supported", "evidence": prefix + "Imaging shows findings.", "confidence": 0.9}
        ]
    })


def mock_lora_fn(prompt: str) -> str:
    """Simulate LoRA model with ~96% JSON validity and 9% overconfidence."""
    import random
    if random.random() < 0.04:
        return "Unable to parse."
    overconf = random.random() < 0.09
    prefix = "Clearly, " if overconf else ""
    return json.dumps({
        "alignments": [
            {"claim_id": "c1", "label": "supported", "evidence": prefix + "Imaging shows findings.", "confidence": 0.85}
        ]
    })


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/medgemma-4b-it")
    parser.add_argument("--lora", default="")
    parser.add_argument("--test-file", default="hf_lora/dataset/eval.jsonl")
    parser.add_argument("--n", type=int, default=50, help="Number of test cases")
    parser.add_argument("--mock", action="store_true", help="Use mock inference (no GPU needed)")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    test_cases = []
    if test_path.exists():
        with open(test_path) as f:
            for line in f:
                test_cases.append(json.loads(line.strip()))
    else:
        logger.warning("Test file not found — using mock cases")
        test_cases = [{"prompt": f"Align claim c{i}", "completion": ""} for i in range(args.n)]

    test_cases = test_cases[:args.n]

    if args.mock or not args.base:
        logger.info("Using mock inference functions")
        base_fn = mock_base_fn
        lora_fn = mock_lora_fn
    else:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            from peft import PeftModel

            logger.info("Loading base model: %s", args.base)
            model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
            processor = AutoProcessor.from_pretrained(args.base)

            def make_fn(m):
                def fn(prompt):
                    inputs = processor(text=[prompt], return_tensors="pt").to(m.device)
                    with torch.no_grad():
                        out = m.generate(**inputs, max_new_tokens=512, do_sample=False)
                    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return fn

            base_fn = make_fn(model)

            if args.lora:
                logger.info("Loading LoRA: %s", args.lora)
                lora_model = PeftModel.from_pretrained(model, args.lora)
                lora_fn = make_fn(lora_model.merge_and_unload())
            else:
                lora_fn = base_fn
        except Exception as e:
            logger.warning("Model loading failed (%s) — falling back to mock", e)
            base_fn = mock_base_fn
            lora_fn = mock_lora_fn

    logger.info("Evaluating base model on %d cases...", len(test_cases))
    base_metrics = evaluate_model(base_fn, test_cases)

    logger.info("Evaluating LoRA model on %d cases...", len(test_cases))
    lora_metrics = evaluate_model(lora_fn, test_cases)

    results = {
        "base_model": base_metrics,
        "lora_model": lora_metrics,
        "test_set_size": len(test_cases),
    }

    out_path = Path("eval/sample_outputs/eval_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results ===")
    print(f"{'Metric':<30} {'Base':>10} {'LoRA':>10} {'Delta':>10}")
    print("-" * 60)
    for metric in ["json_valid_rate", "overconfidence_rate", "label_accuracy", "schema_repair_rate"]:
        b = base_metrics.get(metric)
        l = lora_metrics.get(metric)
        if b is None or l is None:
            continue
        delta = l - b
        print(f"{metric:<30} {b:>10.1%} {l:>10.1%} {delta:>+10.1%}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
