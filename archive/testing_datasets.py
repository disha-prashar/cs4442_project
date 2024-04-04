'''
This program was created to experiment fine-tuning a model with a dataset from HuggingFace.
'''
from transformers import pipeline, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig, TrainingArguments, TextStreamer, DataCollatorWithPadding, Trainer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
import numpy as np
import evaluate

# Load Dataset
raw_datasets = load_dataset("amaydle/npc-dialogue") # Type is DatasetDict

local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/mistral_model"

# Load Base Model (Mistral 7B)
model_name="mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
# model = MistralForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
model = AutoModelForSequenceClassification.from_pretrained(local_model_directory, quantization_config=bnb_config, device_map={"": 0})
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, src_lang="en")
tokenizer = AutoTokenizer.from_pretrained(local_model_directory, use_fast=False, src_lang="en")
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(data):
    return tokenizer(
        data["Name"], data["Biography"], data["Query"], data["Response"], data["Emotion"], padding = "max_length", max_length=1024, truncation=True
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets.column_names)

# tokenized_datasets.with_format("torch")
data_collator = DataCollatorWithPadding(tokenizer)

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
model.config.pad_token_id = model.config.eos_token_id

metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# Training Arguments
# Hyperparameters should be adjusted based on the hardware you using
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps= 2,
    gradient_checkpointing=True,
    # optim = "paged_adamw_8bit",
    # save_steps= 5000,
    # logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    # fp16= False,
    # bf16= False,
    max_grad_norm= 0.3,
    # max_steps= -1,
    # warmup_ratio= 0.3,
    # group_by_length= True,
    # lr_scheduler_type= "constant",
    report_to="wandb",
    # evaluation_strategy="epoch"
)
# Setting sft parameters
trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    # peft_config=peft_config,
    # max_seq_length=1024,
    # dataset_text_field="Biography",
    tokenizer=tokenizer,
    args=training_arguments,
    # packing= False
)

model.resize_token_embeddings(len(tokenizer))

trainer.train()

# Save the fine-tuned model
trainer.save_model(local_model_directory)
# trainer.model.save_pretrained(local_model_directory)
wandb.finish()
# model.config.use_cache = True
# model.eval()


# # Test Model
# def stream(user_prompt):
#     runtimeFlag = "cuda:0"
#     system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
#     B_INST, E_INST = "[INST]", "[/INST]"

#     prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

#     inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

# stream("Explain large language models")

# # Clear the memory footprint
# del model, trainer
# torch.cuda.empty_cache()



# # # Format dataset
# # dataset = dataset.map(
# #     chatml_format,
# #     #remove_columns=original_columns
# # )



# # # def chatml_format(example):

# # #     if len(example['system']) > 0:
# # #         message = {"role": "system", "content": example['system']}
# # #         system = tokenizer.apply_chat_template([message], tokenize=False)
# # #     else:
# # #         system = ""

# # #     message = {"role": "user", "content": example['question']}
# # #     prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
# # #      # Format chosen answer
# # #     chosen = example['chosen'] + "<|im_end|>\n"

# # #     # Format rejected answer
# # #     rejected = example['rejected'] + "<|im_end|>\n"

# # #     return {
# # #         "prompt": system + prompt,
# # #         "chosen": chosen,
# # #         "rejected": rejected,
# # #     }