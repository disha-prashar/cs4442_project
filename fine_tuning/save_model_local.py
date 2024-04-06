'''
This program was used to download and store a model locally from the HuggingFace cloud.
Caution: it will overwrite the current contents of the directory specified by local_model_directory and remove all fine-tuning!
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Load Base Model (Mistral 7B)
model_name="mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, src_lang="en")
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

model.resize_token_embeddings(len(tokenizer))

local_model_directory = "./zephyr_model"
model.save_pretrained(local_model_directory)
tokenizer.save_pretrained(local_model_directory)

# ----------------------------------------------------------------------------------
# model_name="mistralai/Mistral-7B-v0.1"

# model.from_pretrained(local_model_directory)
# tokenizer.from_pretrained(local_model_directory)