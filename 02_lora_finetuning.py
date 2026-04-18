"""
Script 2: Parameter-Efficient Fine-Tuning with LoRA
====================================================
Demonstrates: transformers, PEFT, LoRA adapters, and HuggingFace Trainer.

Fine-tunes a small causal LM on a custom instruction dataset using
Low-Rank Adaptation (LoRA). LoRA trains ~0.1% of the parameters while
matching full fine-tune quality, so it runs on a single consumer GPU.

Dependencies:
    pip install transformers peft datasets accelerate bitsandbytes torch
"""

from __future__ import annotations

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "sshleifer/tiny-gpt2"  # swap for Llama/Qwen/Mistral in practice


def build_dataset(tokenizer) -> Dataset:
    """Build a tiny instruction-tuning dataset. Replace with your own data."""
    examples = [
        {"instruction": "Translate to French: Hello, how are you?",
         "response": "Bonjour, comment allez-vous?"},
        {"instruction": "Summarize: The cat sat on the mat.",
         "response": "A cat is on a mat."},
        {"instruction": "Capital of Japan?", "response": "Tokyo."},
        {"instruction": "2 + 2 = ?", "response": "4."},
    ] * 25  # repeat so the demo has enough steps

    def format_example(ex: dict) -> dict:
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
        tokens = tokenizer(prompt, truncation=True, max_length=128, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return Dataset.from_list(examples).map(format_example, remove_columns=["instruction", "response"])


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # LoRA: inject low-rank trainable matrices into attention projections
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                  # rank of the update matrices
        lora_alpha=16,        # scaling factor
        lora_dropout=0.05,
        target_modules=["c_attn"],  # model-specific; for Llama use ["q_proj","v_proj"]
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="./lora_ckpt",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    # Save only the adapter weights (tiny — a few MB)
    model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")
    print("\nLoRA adapter saved to ./lora_adapter")

    # Quick inference check
    model.eval()
    prompt = "### Instruction:\nCapital of Japan?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Sample generation:", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
