"""
Generate synthetic training pairs for RTL LoRA adapters.

Creates two datasets:
1. JSON schema compliance: (prompt, bad_output) → (prompt, good_output)
2. Uncertainty calibration: (report_text) → (calibrated_labels)

Uses MedGemma itself (via mock data) + hand-crafted augmentations.
All data is synthetic — no PHI.
"""
import json
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

CLAIM_TEMPLATES = [
    ("There is {finding} in the {location}.", "finding"),
    ("No {finding} is identified.", "absence"),
    ("The {finding} measures {size} cm.", "measurement"),
    ("Findings are consistent with {diagnosis}.", "impression"),
    ("{finding} is noted, possibly representing {diagnosis}.", "impression"),
    ("Mild {finding} is present.", "finding"),
    ("The {location} appears within normal limits.", "finding"),
]

FINDINGS = ["consolidation", "opacity", "effusion", "atelectasis", "pneumothorax",
            "infiltrate", "nodule", "mass", "cardiomegaly", "hyperinflation"]
LOCATIONS = ["right lower lobe", "left upper lobe", "bilateral lung bases",
             "right hemithorax", "left costophrenic angle", "mediastinum", "right hilum"]
DIAGNOSES = ["pneumonia", "heart failure", "COPD", "pulmonary edema", "lung cancer",
             "pleural effusion", "atelectasis"]
SIZES = [str(round(random.uniform(0.5, 4.0), 1)) for _ in range(20)]


def make_claim(i: int) -> dict:
    template, ctype = random.choice(CLAIM_TEMPLATES)
    text = template.format(
        finding=random.choice(FINDINGS),
        location=random.choice(LOCATIONS),
        size=random.choice(SIZES),
        diagnosis=random.choice(DIAGNOSES),
    )
    return {"claim_id": f"c{i+1}", "text": text, "sentence_span": {"start": i*60, "end": i*60+len(text)}, "claim_type": ctype}


LABELS = ["supported", "uncertain", "needs_review"]

OVERCONFIDENT_PHRASES = [
    ("There is definitely", "There appears to be"),
    ("This is consistent with", "Findings may be consistent with"),
    ("Clearly shows", "Suggests"),
    ("No doubt", "Possibly"),
    ("Confirms", "May suggest"),
]


def make_alignment_example(n_claims: int = 4) -> dict:
    claims = [make_claim(i) for i in range(n_claims)]
    alignments = []
    for claim in claims:
        label = random.choices(LABELS, weights=[0.5, 0.3, 0.2])[0]
        alignments.append({
            "claim_id": claim["claim_id"],
            "label": label,
            "evidence": f"Visual evidence {'supports' if label=='supported' else 'does not clearly support'} this claim.",
            "confidence": round(random.uniform(0.5, 0.95), 2),
            "related_finding_ids": [f"f{random.randint(1,3)}"],
            "claim_text": claim["text"],
        })
    return {"claims": claims, "alignments": alignments}


def make_json_compliance_pair() -> dict:
    """Return (prompt, bad_response, good_response) for JSON schema compliance training."""
    example = make_alignment_example()
    claims_json = json.dumps(example["claims"], indent=2)
    schema = json.dumps({
        "type": "object", "required": ["alignments"],
        "properties": {"alignments": {"type": "array"}}
    })
    prompt = f"Align the following claims to image findings.\nClaims:\n{claims_json}\nRespond with JSON matching: {schema}"
    good = json.dumps({"alignments": example["alignments"]}, indent=2)
    # Simulate bad output (no JSON, or truncated)
    bad = random.choice([
        "The claims appear to be supported by the imaging evidence.",
        "```alignment\n" + good[:50],
        '{"partial": true}',
    ])
    return {"prompt": prompt, "bad": bad, "good": good}


def make_uncertainty_pair() -> dict:
    """Return (overconfident_text, calibrated_text) for uncertainty training."""
    # Randomly select a claim text and make it overconfident, then calibrate
    claim_text = make_claim(0)["text"]
    original = claim_text
    calibrated = claim_text
    for over, cal in OVERCONFIDENT_PHRASES:
        if over.lower() in original.lower():
            calibrated = original.replace(over, cal)
            break
    # Add a random overconfident phrase
    original = random.choice(["Definite", "Clearly", "Obviously", ""]) + " " + original
    original = original.strip()
    return {"overconfident": original, "calibrated": calibrated}


def generate_dataset(n_train: int = 200, n_eval: int = 50) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON schema compliance dataset
    train_pairs = [make_json_compliance_pair() for _ in range(n_train)]
    eval_pairs = [make_json_compliance_pair() for _ in range(n_eval)]

    train_jsonl = OUTPUT_DIR / "train.jsonl"
    eval_jsonl = OUTPUT_DIR / "eval.jsonl"

    with open(train_jsonl, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps({"prompt": pair["prompt"], "completion": pair["good"]}) + "\n")
    with open(eval_jsonl, "w") as f:
        for pair in eval_pairs:
            f.write(json.dumps({"prompt": pair["prompt"], "completion": pair["good"]}) + "\n")

    # Uncertainty calibration dataset
    unc_train = [make_uncertainty_pair() for _ in range(n_train)]
    unc_eval = [make_uncertainty_pair() for _ in range(n_eval)]

    with open(OUTPUT_DIR / "uncertainty_train.jsonl", "w") as f:
        for pair in unc_train:
            f.write(json.dumps({"input": pair["overconfident"], "output": pair["calibrated"]}) + "\n")
    with open(OUTPUT_DIR / "uncertainty_eval.jsonl", "w") as f:
        for pair in unc_eval:
            f.write(json.dumps({"input": pair["overconfident"], "output": pair["calibrated"]}) + "\n")

    print(f"Generated {n_train} train + {n_eval} eval pairs in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_dataset()
