import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

bucket_name = os.getenv("BUCKET_NAME", "YOUR_BUCKET_NAME")

model_name = os.getenv("MODEL_NAME", "google/gemma-3-1b-it")

# Load your CSV dataset
# Assumes the script is run from the fine-tuning directory, adjust path if needed
csv_path = "./train.csv"
dataset = load_dataset("csv", data_files=csv_path)["train"]

# Format for chat: Prompt as input, Clinician as target
# Setup input formats: trains the model to respond to Prompt with Clinician output.
tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_to_chat(example):
    return {
        "conversations": [
            {"role": "user", "content": example["Prompt"]},
            {"role": "assistant", "content": example["Clinician"]},
        ]
    }

formatted_dataset = dataset.map(
    format_to_chat,
    batched=False,                        # Process row by row
    remove_columns=dataset.column_names,  # Keep only the new column
)

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False)
    return {"text": texts}

final_dataset = formatted_dataset.map(apply_chat_template, batched=True)

print(final_dataset)

bnb_4bit_compute_dtype = "float16" 
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,     
    lora_dropout=0.1,  
    r=8,             
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Batch size per GPU for training
    gradient_accumulation_steps=2,  # Number of update steps to accumulate the gradients for
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=5,
    learning_rate=2e-4,    # Initial learning rate (AdamW optimizer)
    weight_decay=0.001,    # Weight decay to apply to all layers except bias/LayerNorm weights
    fp16=True, bf16=False, # Enable fp16/bf16 training
    max_grad_norm=0.3,     # Maximum gradient normal (gradient clipping)
    warmup_ratio=0.03,     # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length=True,  # Group sequences into batches with same length # Saves memory and speeds up training considerably
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,  # Maximum sequence length to use
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,       # Pack multiple short examples in the same input sequence to increase efficiency
)

trainer.train()
trainer.model.save_pretrained(os.getenv("NEW_MODEL", ""))

################################# Save ########################################

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": torch.cuda.current_device()},
)
model = PeftModel.from_pretrained(base_model, os.getenv("NEW_MODEL", "gemma-emoji"))
model = model.merge_and_unload()

# push results to Cloud Storage
file_path_to_save_the_model = "/finetune/new_model"
model.save_pretrained(file_path_to_save_the_model)
tokenizer.save_pretrained(file_path_to_save_the_model)

