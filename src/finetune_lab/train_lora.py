"""Real LoRA/PEFT fine-tuning path — requires GPU + `pip install torch transformers peft datasets`.

Run on Colab T4 (free): see notebooks/colab_t4.md. This module is intentionally import-guarded so
the offline pipeline and CI never require torch.

The flow mirrors what you would do with a real base model:
  1. load instruction records produced by finetune_lab.data.to_instruction_format
  2. attach LoRA adapters (r=16, alpha=32, target q/k/v/o projections)
  3. SFT with packing off, bf16 on T4->fp16
  4. merge adapters, export, evaluate against the held-out split with the same micro_f1 metric
"""

from __future__ import annotations

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "bf16": False,
    "fp16": True,
    "logging_steps": 10,
}


def peft_available() -> bool:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def build_training_script(model_name: str, dataset_path: str, output_dir: str) -> str:
    lora = DEFAULT_LORA_CONFIG
    args = TRAINING_ARGS
    return f'''from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

base = "{model_name}"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
lora = LoraConfig(r={lora["r"]}, lora_alpha={lora["lora_alpha"]}, lora_dropout={lora["lora_dropout"]},
                  target_modules={lora["target_modules"]})
model = get_peft_model(model, lora)
model.print_trainable_parameters()

ds = load_dataset("json", data_files="{dataset_path}", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tok,
    args=TrainingArguments(output_dir="{output_dir}",
                           num_train_epochs={args["num_train_epochs"]},
                           per_device_train_batch_size={args["per_device_train_batch_size"]},
                           gradient_accumulation_steps={args["gradient_accumulation_steps"]},
                           learning_rate={args["learning_rate"]},
                           lr_scheduler_type="{args["lr_scheduler_type"]}",
                           warmup_ratio={args["warmup_ratio"]},
                           fp16=True, logging_steps=10),
    dataset_text_field=None, max_seq_length=1024,
)
trainer.train()
trainer.save_model("{output_dir}")
'''
