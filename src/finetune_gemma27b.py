# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script fine-tunes the Gemma-3-27b-it model for pathfinding in maps.

It uses the SFTTrainer from the TRL library to perform supervised fine-tuning.
The script loads a training dataset of map images and corresponding paths,
formats the data into a conversational format, and then trains the model.
"""
import argparse
import os
import json
import random
import pandas as pd
import io
import re
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig, get_peft_model
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser(
    description="Fine-tuning code for the pathfinding model.")
parser.add_argument(
    "--model_id",
    default="google/gemma3-27b-it",
    required=True,
    help="The model ID to use for fine-tuning.")
parser.add_argument(
    "--output_dir",
    required=True,
    help="The output directory to save the fine-tuned model.")
parser.add_argument(
    "--train_path_dir",
    required=True,
    help="The directory containing the training data.")
args = parser.parse_args()


model_id = args.model_id
output_dir = args.output_dir
train_path_dir = args.train_path_dir

def format_data(samples):
    img = Image.open(io.BytesIO(samples['image']))
    samples['images'] = [img]
    samples['messages'][0]['content'][0] = {"type": "image", "image": img}
    return samples

train_folders = [os.path.join(train_path_dir, f) for f in os.listdir(train_path_dir) if f.endswith('.parquet')]
print("Len train folders ", len(train_folders))
data = load_dataset(
        "parquet",
        data_files=train_folders,
        split='train',
        streaming=True
)
data = data.map(
    format_data,
    remove_columns=['image']
)

# Load model and tokenizer
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id, do_pan_and_scan=True)


# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['gate_proj', 'down_proj', 'v_proj', 'k_proj', 'q_proj', 'o_proj', 'up_proj'],
)

# Configure training arguments
training_args = SFTConfig(
    output_dir=output_dir,  # Directory to save the model
    max_steps=20_000,
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=1,  # Steps to accumulate gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    max_length=2048,
    dataloader_num_workers=8,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=5e-7,  # Learning rate for training
    # Logging and evaluation
    logging_steps=50,  # Steps interval for logging
    save_strategy="steps",  # Strategy for saving the model
    save_steps=500,  # Steps interval for saving
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="none",  # Reporting tool for tracking metrics
    dataloader_drop_last=True,
    accelerator_config={
        "dispatch_batches": False
    }
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    peft_config=peft_config,
    processing_class=processor,
)

if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trainer.train()

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(training_args.output_dir)
