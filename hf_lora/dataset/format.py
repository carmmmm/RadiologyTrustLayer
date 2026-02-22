"""Format synthetic JSONL pairs into chat-template format for MedGemma fine-tuning."""
import json
from pathlib import Path


def format_for_chat(input_path: Path, output_path: Path, model_id: str = "google/medgemma-4b-it") -> None:
    """
    Convert prompt/completion pairs to Gemma chat template format.
    Output: JSONL with {"text": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            pair = json.loads(line.strip())
            prompt = pair.get("prompt", pair.get("input", ""))
            completion = pair.get("completion", pair.get("output", ""))
            text = (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n{completion}<end_of_turn>"
            )
            fout.write(json.dumps({"text": text}) + "\n")
            count += 1
    print(f"Formatted {count} pairs → {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent
    for split in ["train", "eval"]:
        format_for_chat(base / f"{split}.jsonl", base / f"{split}_chat.jsonl")
    for split in ["train", "eval"]:
        format_for_chat(base / f"uncertainty_{split}.jsonl", base / f"uncertainty_{split}_chat.jsonl")
