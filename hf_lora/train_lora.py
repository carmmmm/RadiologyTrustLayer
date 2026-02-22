"""
RTL LoRA training script using TRL + PEFT.

Trains a LoRA adapter on MedGemma-4B-IT to improve:
1. JSON schema compliance (output structure reliability)
2. Calibrated uncertainty language (reduced overconfidence)

Usage:
  python hf_lora/train_lora.py --config hf_lora/configs/lora_json.schema.yaml

Requirements:
  pip install peft trl transformers accelerate datasets bitsandbytes
"""
import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments
    from trl import SFTTrainer

    model_id = cfg["model_id"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    output_cfg = cfg["output"]

    logger.info("Loading base model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["eval_file"],
        },
    )

    training_args = TrainingArguments(
        output_dir=output_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        packing=False,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_cfg["output_dir"])
    logger.info("Training complete. Model saved to %s", output_cfg["output_dir"])

    # Save training summary
    summary = {
        "model_id": model_id,
        "lora_config": lora_cfg,
        "train_samples": len(dataset["train"]),
        "eval_samples": len(dataset["validation"]),
        "output_dir": output_cfg["output_dir"],
    }
    Path(output_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(output_cfg["output_dir"]) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train RTL LoRA adapter")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    if args.dry_run:
        logger.info("Dry run — config is valid. Exiting.")
        return

    # Generate synthetic data if not present
    train_path = Path(cfg["data"]["train_file"])
    if not train_path.exists():
        logger.info("Training data not found — generating synthetic dataset...")
        from hf_lora.dataset.make_synthetic import generate_dataset
        from hf_lora.dataset.format import format_for_chat
        generate_dataset()
        format_for_chat(train_path, train_path.parent / (train_path.stem + "_chat.jsonl"))
        cfg["data"]["train_file"] = str(train_path.parent / (train_path.stem + "_chat.jsonl"))
        eval_path = Path(cfg["data"]["eval_file"])
        format_for_chat(eval_path, eval_path.parent / (eval_path.stem + "_chat.jsonl"))
        cfg["data"]["eval_file"] = str(eval_path.parent / (eval_path.stem + "_chat.jsonl"))

    train(cfg)


if __name__ == "__main__":
    main()
