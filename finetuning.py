import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
import itertools


model_name = "/pre-model-full-multi/final"
data_path = "/data/llama3_sft_data_new-v3.jsonl"
output_base_dir = "/fine_tuning/qlora_grid_search"

os.makedirs(output_base_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


param_grid = {
    "learning_rate": [8e-5,1e-4],
    "per_device_train_batch_size": [4],
    "num_train_epochs": [4],
    "lora_dropout": [0.05],
    "seed": [45,123],
    "num_layers_to_finetune": [32,24,15],
}

param_combinations = list(itertools.product(*param_grid.values()))
keys = list(param_grid.keys())

print("🔧 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

def messages_to_text(messages):
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            text += f"System: {content}\n"
        elif role == "user":
            text += f"User: {content}\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n"
    return text.strip()


raw_data = Dataset.from_json(data_path)
text_data = raw_data.map(
    lambda ex: {"text": messages_to_text(ex["messages"])},
    remove_columns=["messages"]
)

train_test_split = text_data.train_test_split(test_size=0.2)


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=1024,
        add_special_tokens=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print(" Tokenizing dataset...")
tokenized_train = train_test_split["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)
tokenized_val = train_test_split["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)


def get_target_modules(model, num_layers):

    import torch.nn as nn

    total_layers = model.config.num_hidden_layers
    start_layer = max(0, total_layers - num_layers)
    target_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                try:
                    layer_idx = int(parts[2])
                    if layer_idx >= start_layer:
                        target_modules.add(name)
                except (ValueError, IndexError):
                    continue

    if not target_modules:
        raise ValueError(f"×")
    
    return list(target_modules)


for i, params in enumerate(param_combinations):
    config = dict(zip(keys, params))
    print(f"\n [{i+1}/{len(param_combinations)}] 配置: {config}")

    model = None
    trainer = None

    try:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )


        target_modules = get_target_modules(model, config["num_layers_to_finetune"])

        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=target_modules,
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        lr = config["learning_rate"]
        layers = config["num_layers_to_finetune"]
        dropout = config["lora_dropout"]
        seed = config["seed"]
        batch = config["per_device_train_batch_size"]
        epochs = config["num_train_epochs"]
        

        lr_str = f"{lr:.0e}".replace("e-0", "e-")
        
        output_dir_name = f"lr{lr_str}_layers{layers}_drop{dropout:.2f}_seed{seed}_batch{batch}_epochs{epochs}"
        output_dir = os.path.join(output_base_dir, output_dir_name)


        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_train_batch_size"],
            learning_rate=config["learning_rate"],
            num_train_epochs=config["num_train_epochs"],
            logging_steps=50,
            eval_steps=100,
            save_steps=100,
            eval_strategy="steps",       
            save_total_limit=None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            optim="adamw_torch",
            seed=config["seed"],
            report_to="none",
            fp16=False,
            bf16=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            peft_config=peft_config,
        )


        trainer.train()

        final_model_dir = os.path.join(output_dir, "final_adapter")
        trainer.model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        eval_result = trainer.evaluate()
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"Config: {config}\n")
            f.write(f"Eval Loss: {eval_result['eval_loss']:.6f}\n")
            f.write(f"Target Modules Count: {len(target_modules)}\n")


    except Exception as e:
        error_file = os.path.join(output_base_dir, f"error_run_{i}.txt")
        with open(error_file, "w") as f:
            f.write(f"Config: {config}\nError: {str(e)}\n")

    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        torch.cuda.empty_cache()
