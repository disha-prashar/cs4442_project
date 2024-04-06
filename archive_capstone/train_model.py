'''
This file was used for initial exploration and experimentation with fine-tuning an AI model.
A pre-existing dataset on the HuggingFace community was used for fine-tuning.
'''

from transformers import pipeline, AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig, TrainingArguments, TextStreamer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings

# Load Dataset
train_dataset = load_dataset("amaydle/npc-dialogue", split="train")
test_dataset = load_dataset("amaydle/npc-dialogue", split="test")
# print(train_dataset)
# print(test_dataset)

# local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/downloaded_model/Mistral-7B-Instruct-v0.1"
local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"

# Load Base Model (Mistral 7B)
model_name="mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(local_model_directory, local_files_only = True, quantization_config=bnb_config, device_map={"": 0}) # .to('cuda')
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_directory, src_lang="en")
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set up Model Training Tracking and Visualization
wandb.login(key = "ea369874684d6636d5a86e352b7968d17436787b")
run = wandb.init(project='Fine tuning zephyr 7B', job_type="training", anonymous="allow")

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

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['Name'])):
#         # text = f"### Name: {example['Name'][i]}\n ### Biography: {example['Biography'][i]}\n ### Query: {example['Query'][i]}\n ### Response: {example['Response'][i]}\n ### Emotion: {example['Emotion'][i]}"
#         # text = f"Response: {}\n"
#         output_texts.append(text)
#     return output_texts

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=1024,
    dataset_text_field="Biography",
    tokenizer=tokenizer,
    args=training_arguments,
    # formatting_func=formatting_prompts_func,
    # packing= True,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(local_model_directory)
# model.save_pretrained(local_model_directory)
# tokenizer.save_pretrained(local_model_directory)
# trainer.model.save_pretrained(save_directory)
wandb.finish()
model.config.use_cache = True
model.eval()


'''
# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, tokenizer, examples):
        self.examples = examples
        self.tokenizer = tokenizer
        # Set the sep_token explicitly
        if tokenizer.sep_token is None:
            self.tokenizer.sep_token = "[SEP]"

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        inputs = self.tokenizer(example["Title"] + self.tokenizer.sep_token + example["Objective"], padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(example["Text"], padding="max_length", truncation=True, return_tensors="pt")
        return inputs, labels
'''


# # # Collate function to prepare batch inputs and labels
# # def data_collator(tokenizer: PreTrainedTokenizer, examples: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
# #     combined_inputs = []
# #     labels = []

# #     for example in examples:
# #         # Check if required fields are present in the example
# #         if "Title" not in example or "Objective" not in example or "Text" not in example:
# #             raise ValueError("Missing required fields in the example.")

# #         # Concatenate Title and Objective columns
# #         combined_input = example["Title"] + tokenizer.sep_token + example["Objective"]
# #         print(combined_input)
# #         combined_inputs.append(combined_input)

# #         # Collect responses (labels)
# #         labels.append(example["Text"])

# #     # Tokenize combined inputs
# #     tokenized_inputs = tokenizer(combined_inputs, padding=True, truncation=True, return_tensors="pt")

# #     # Tokenize responses
# #     tokenized_labels = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
    
# #     return {"input_ids": tokenized_inputs.input_ids,
# #             "attention_mask": tokenized_inputs.attention_mask,
# #             "labels": tokenized_labels.input_ids}

