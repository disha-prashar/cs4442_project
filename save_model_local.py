'''
This program was used to download and store a model locally from the HuggingFace cloud.
Caution: it will overwrite the current contents of the directory specified by local_model_directory and remove all fine-tuning!
'''
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os, torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

# Load Base Model (Mistral 7B)
model_name="mistralai/Mistral-7B-v0.1"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit= True,
#     bnb_4bit_quant_type= "nf4",
#     bnb_4bit_compute_dtype= torch.bfloat16,
#     bnb_4bit_use_double_quant= False,
# )
model = AutoModelForCausalLM.from_pretrained(model_name) #, quantization_config=bnb_config, device_map={"": 0}
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

local_model_directory = "./local_model"
model.save_pretrained(local_model_directory)
tokenizer.save_pretrained(local_model_directory)

# ----------------------------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModel

# model_name="mistralai/Mistral-7B-v0.1"

# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/mistral_model"
# model.save_pretrained(local_model_directory)
# tokenizer.save_pretrained(local_model_directory)

# model.from_pretrained(local_model_directory)
# tokenizer.from_pretrained(local_model_directory)