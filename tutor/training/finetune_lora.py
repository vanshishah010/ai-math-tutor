"""LoRA/QLoRA Fine-Tuning Script for Qwen2.5-Math-1.5B-Instruct.

Supports training on free GPU environments (Google Colab T4 / Kaggle P100/T4x2)
using Hugging Face transformers, peft, and trl.
Auto-selects precision (bf16 vs fp16) based on GPU capabilities, calculates dynamic
evaluation steps (~10 per run), logs losses to CSV, and saves adapters to models/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(config_path: str = "tutor/training/config.yaml") -> Dict[str, Any]:
    """Load training hyperparameters from YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chat_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset containing ChatML messages."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def train_lora(
    config_path: str = "tutor/training/config.yaml",
    dry_run: bool = False,
    override_output_dir: Optional[str] = None,
    override_epochs: Optional[int] = None,
    limit: Optional[int] = None,
    loss_log_path: str = "tutor/training/logs/loss_log.csv",
) -> str:
    """Run QLoRA fine-tuning on a small math reasoning LLM.

    Args:
        config_path: Path to configuration YAML file.
        dry_run: If True, validates config and data without loading model weights.
        override_output_dir: Custom output path for adapter weights.
        override_epochs: Custom number of training epochs.
        limit: Maximum number of training samples to load.
        loss_log_path: Path to write training and evaluation loss logs.

    Returns:
        Path where adapter was saved (or output directory if dry-run).
    """
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    output_dir = override_output_dir or train_cfg.get("output_dir", "models/qwen2.5-math-tutor-lora")
    epochs = override_epochs or train_cfg.get("num_train_epochs", 1)

    train_path = train_cfg.get("dataset_train_path", "tutor/data/processed/train.jsonl")
    val_path = train_cfg.get("dataset_val_path", "tutor/data/processed/val.jsonl")

    logger.info("Initializing LoRA fine-tuning for base model: %s", model_cfg.get("base_model_name"))
    logger.info("Output directory: %s", output_dir)
    logger.info("Target epochs: %d", epochs)

    # Validate dataset existence
    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{train_path}'. "
            f"Please run `python tutor/data/build_dataset.py` first."
        )

    train_data = load_chat_dataset(train_path)
    if limit is not None and limit > 0:
        train_data = train_data[:limit]
        logger.info("Training dataset limited to %d samples via --limit", len(train_data))

    val_data = load_chat_dataset(val_path) if Path(val_path).exists() else []
    if limit is not None and limit > 0 and val_data:
        val_limit = min(len(val_data), max(10, limit // 10))
        val_data = val_data[:val_limit]

    logger.info("Loaded %d training samples and %d validation samples", len(train_data), len(val_data))

    # Calculate dynamic evaluation and saving steps (~10 evaluations per run, never 0)
    batch_size = train_cfg.get("per_device_train_batch_size", 2)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 8)
    effective_batch_size = max(1, batch_size * grad_accum)
    total_training_steps = max(1, (len(train_data) * epochs) // effective_batch_size)
    step_interval = max(1, total_training_steps // 10)
    save_steps = step_interval
    eval_steps = step_interval

    logger.info(
        "Training schedule: %d total optimization steps (effective batch size: %d). "
        "Dynamic save_steps=%d, eval_steps=%d (~10 evaluations per run).",
        total_training_steps, effective_batch_size, save_steps, eval_steps
    )

    # Ensure logs directory exists
    Path(loss_log_path).parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[Dry Run] Config, schedule, and datasets validated successfully.")
        logger.info("[Dry Run] Target steps: %d | eval_steps: %d | save_steps: %d", total_training_steps, eval_steps, save_steps)
        logger.info("[Dry Run] Loss log destination: %s", loss_log_path)
        logger.info("[Dry Run] Skipping heavy model download and training execution.")
        return output_dir

    # Optional torch / transformers imports for training execution environments
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise ImportError(
            f"Required deep learning libraries not found: {e}\n"
            f"Please install: pip install -r requirements-train.txt"
        )

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    logger.info("Training device: %s (CUDA Available: %s)", device_name, has_cuda)

    # Auto-select precision: bf16 only if hardware supports it, otherwise fp16
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    if supports_bf16:
        use_bf16 = True
        use_fp16 = False
        compute_dtype = torch.bfloat16
        model_dtype = torch.bfloat16
        precision_str = "bfloat16 (Ampere / Ada / Hopper / TPU supported)"
    elif has_cuda:
        use_bf16 = False
        use_fp16 = True
        compute_dtype = torch.float16
        model_dtype = torch.float16
        precision_str = "float16 (T4 / P100 / Volta GPU fallback, bf16 unsupported)"
    else:
        use_bf16 = False
        use_fp16 = False
        compute_dtype = torch.float32
        model_dtype = torch.float32
        precision_str = "float32 (CPU execution)"

    logger.info("Precision configuration: %s", precision_str)

    # 1. Quantization Configuration (QLoRA)
    bnb_config = None
    if has_cuda and model_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )

    # 2. Tokenizer
    base_model = model_cfg.get("base_model_name", "Qwen/Qwen2.5-Math-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto") if has_cuda else None,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    if has_cuda and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA PEFT Configuration
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Format datasets for SFTTrainer
    def format_chat_prompt(example: Dict[str, Any]) -> Dict[str, str]:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    hf_train_dataset = Dataset.from_list(train_data).map(format_chat_prompt)
    hf_val_dataset = Dataset.from_list(val_data).map(format_chat_prompt) if val_data else None

    # 6. Loss Logging Callback
    class LossLoggingCallback(TrainerCallback):
        """Logs training and evaluation loss metrics to a CSV file."""

        def __init__(self, log_path: str) -> None:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "epoch", "loss", "eval_loss", "learning_rate"])

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            epoch = round(state.epoch, 4) if state.epoch is not None else ""
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")

            if loss is not None or eval_loss is not None:
                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, epoch, loss or "", eval_loss or "", lr or ""])

    # 7. SFTConfig (trl>=0.17.0)
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=train_cfg.get("max_seq_length", 512),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=max(1, min(step_interval, 10)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 2),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if hf_val_dataset else "no",
        eval_steps=eval_steps if hf_val_dataset else None,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=train_cfg.get("optim", "paged_adamw_8bit") if has_cuda else "adamw_torch",
        report_to="none",
        seed=train_cfg.get("seed", 42),
    )

    # 8. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[LossLoggingCallback(loss_log_path)],
    )

    logger.info("Starting model fine-tuning...")
    trainer.train()

    # 9. Save adapter and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA fine-tuning complete! Adapter saved to: %s", output_dir)
    logger.info("Loss log recorded at: %s", loss_log_path)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Math LLM with LoRA/QLoRA")
    parser.add_argument("--config", type=str, default="tutor/training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of training samples")
    parser.add_argument("--loss-log", type=str, default="tutor/training/logs/loss_log.csv", help="Path to loss CSV log")

    args = parser.parse_args()
    train_lora(
        config_path=args.config,
        dry_run=args.dry_run,
        override_output_dir=args.output_dir,
        override_epochs=args.epochs,
        limit=args.limit,
        loss_log_path=args.loss_log,
    )


if __name__ == "__main__":
    main()
