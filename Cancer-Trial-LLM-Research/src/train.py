import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# QLoRA and training configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
output_dir = "./experiments/results"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
fp16 = True

device_map = {"": 0}

# Create bitsandbytes config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=fp16,
    max_grad_norm=1.0,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

# Load dataset (assumed to be saved as a pickle file)
from datasets import Dataset
import pandas as pd
train_data = pd.read_pickle("data/train_data_llama2.pkl")
dataset = Dataset.from_pandas(train_data[['llama2_message']])

# Initialize trainer and train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field='llama2_message',
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)
trainer.train(resume_from_checkpoint=True)
