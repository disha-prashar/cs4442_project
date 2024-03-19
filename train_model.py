from transformers import pipeline, AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig, TrainingArguments, TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings

# Load Dataset
train_dataset = load_dataset("amaydle/npc-dialogue", split="train")
test_dataset = load_dataset("amaydle/npc-dialogue", split="test")
print(train_dataset)
print(test_dataset)

# Load Base Model (Mistral 7B)
model_name="mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = MistralForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0}) # .to('cuda')
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en")
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set up Model Training Tracking and Visualization
wandb.login(key = "ea369874684d6636d5a86e352b7968d17436787b")
run = wandb.init(project='Fine tuning mistral 7B', job_type="training", anonymous="allow")

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

# Training Arguments
# Hyperparameters should be adjusted based on the hardware you using
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 5000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="Biography",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

# Save the fine-tuned model
local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/local_model_2"
trainer.model.save_pretrained(local_model_directory)
wandb.finish()
model.config.use_cache = True
model.eval()
