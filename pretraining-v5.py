#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full continued pretraining of LLaMA-3 8B on multi-A100.
Supports:
  - Single .txt / .jsonl file via --data_file
  - Directory of .md files via --data_dir (recursive)
"""

import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

HF_CACHE_DIR = "/data2/wanghao/large"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    import markdown
    from bs4 import BeautifulSoup
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    print("Warning: 'markdown' or 'beautifulsoup4' not installed. Markdown files will be used as raw text.")

def md_to_text(md: str) -> str:
    """Convert Markdown to plain text."""
    if not HAS_MARKDOWN:
        return md
    try:
        html = markdown.markdown(md)
        return BeautifulSoup(html, "html.parser").get_text()
    except Exception:
        return md  # fallback

def load_text_files(file_paths: List[str], convert_md: bool = True) -> List[Dict[str, str]]:
    """Load text from files. If .md and convert_md=True, convert to plain text."""
    examples = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            if convert_md and fp.lower().endswith(".md"):
                text = md_to_text(text)
            examples.append({"text": text})
        except Exception as e:
            print(f"Skipping {fp} due to error: {e}")
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/Meta-Llama-3-8B-Instruct-model")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to a single .txt or .jsonl file (optional)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing .md files (optional, recursive)")
    parser.add_argument("--output_dir", type=str, default="/pretraining/pre-model-full-multi")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--min_token_length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    args = parser.parse_args()

    if not args.data_file and not args.data_dir:
        raise ValueError("Please provide either --data_file or --data_dir.")

    # === Load all data into a list of {'text': ...} ===
    all_examples = []

    if args.data_file:
        from datasets import load_dataset
        if args.data_file.endswith(".jsonl"):
            ds = load_dataset("json", data_files=args.data_file)["train"]
            all_examples.extend(ds)
        else:
            ds = load_dataset("text", data_files=args.data_file)["train"]
            all_examples.extend(ds)

    if args.data_dir:
        md_files = glob.glob(os.path.join(args.data_dir, "**/*.md"), recursive=True)
        print(f"Loading {len(md_files)} .md files from {args.data_dir}")
        md_examples = load_text_files(md_files, convert_md=True)
        all_examples.extend(md_examples)

    if not all_examples:
        raise ValueError("No valid training examples found!")

    dataset = Dataset.from_list(all_examples)
    text_column = "text"


    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0, num_proc=4)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False


    def tokenize_function(examples):
        outputs = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_overflowing_tokens=True,
            return_length=True,
            stride=args.stride,
        )
        input_ids = []
        for ids, length in zip(outputs["input_ids"], outputs["length"]):
            if length >= args.min_token_length:
                input_ids.append(ids)
        return {"input_ids": input_ids}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing & chunking",
    )


    eval_size = min(1000, max(100, len(tokenized_dataset) // 100))
    split_dataset = tokenized_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=False,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=42,
        ddp_find_unused_parameters=False,
        log_on_each_node=False,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
        fsdp_config={
            "limit_all_gathers": True,
            "forward_prefetch": False,
            "backward_prefetch": "BACKWARD_PRE",
            "use_orig_params": True,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


    if trainer.is_world_process_zero():
        total_tokens = sum(len(ids) for ids in tokenized_dataset["input_ids"])
        avg_len = total_tokens / len(tokenized_dataset)
        num_gpus = torch.cuda.device_count()
        total_batch = (
            args.per_device_train_batch_size *
            args.gradient_accumulation_steps *
            num_gpus
        )
        print(f"Starting FULL pretraining on {num_gpus}×A100...")
        print(f"Total samples: {len(tokenized_dataset)}, Avg len: {avg_len:.1f}")
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        print(f"Batch: {total_batch} (per device: {args.per_device_train_batch_size}, accum: {args.gradient_accumulation_steps})")
        print(f"Max seq len: {args.max_seq_length}, Stride: {args.stride}")


    trainer.train()


    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

        

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"✅ Full model saved to: {final_dir}")
        

if __name__ == "__main__":
    main()