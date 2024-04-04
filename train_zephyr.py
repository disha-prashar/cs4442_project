from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch, wandb

# Load Dataset
train_dataset = load_dataset("dprashar/npc_dialogue_rpg_quests", split="train")

# Load Base Model (Zephyr 7B) from local directory and adjust parameters
local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(local_model_directory, local_files_only = True, quantization_config=bnb_config, device_map={"": 0})
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load Tokenizer from local directory and prepare for training
tokenizer = AutoTokenizer.from_pretrained(local_model_directory, src_lang="en")
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepping dataset for training
encoded_dataset = train_dataset.map(lambda e: tokenizer(e['Title'], truncation=True, padding=True), batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Set up Model Training Tracking and Visualization
wandb.login(key = "") # Insert WanDB key here
run = wandb.init(project='Fine tuning Zephyr 7B', job_type="training", anonymous="allow")

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
    overwrite_output_dir=True,
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

# Function to define formatting for prompts during training to be applied when interacting with the model
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["Title"])):
       text = f"### Title: {example['Title'][i]}\n ### Objective: {example['Objective'][i]}\n ### Text: {example['Text'][i]}\n"
       output_texts.append(text)
    return output_texts

# Define Trainer and set sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=encoded_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(local_model_directory)

# Ends reporting to WanDB
wandb.finish()
