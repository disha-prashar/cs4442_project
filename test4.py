from transformers import pipeline, AutoTokenizer, MistralForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, logging, TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings

model_name="mistralai/Mistral-7B-v0.1"
# local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470"
model = MistralForCausalLM.from_pretrained(model_name)
model.to('cuda')

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit= True,
#     bnb_4bit_quant_type= "nf4",
#     bnb_4bit_compute_dtype= torch.bfloat16,
#     bnb_4bit_use_double_quant= False,
# )


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompt = "What is the capital of Australia?"
prompt_template=f'''{prompt}
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to('cuda')
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# dataset = load_dataset("amaydle/npc-dialogue", split='test')

# # Format dataset
# dataset = dataset.map(
#     chatml_format,
#     #remove_columns=original_columns
# )


# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
# )
# model = get_peft_model(model, peft_config)

# training_arguments = TrainingArguments(
#     output_dir= "./results",
#     num_train_epochs= 1,
#     per_device_train_batch_size= 8,
#     gradient_accumulation_steps= 2,
#     optim = "paged_adamw_8bit",
#     save_steps= 5000,
#     logging_steps= 30,
#     learning_rate= 2e-4,
#     weight_decay= 0.001,
#     fp16= False,
#     bf16= False,
#     max_grad_norm= 0.3,
#     max_steps= -1,
#     warmup_ratio= 0.3,
#     group_by_length= True,
#     lr_scheduler_type= "constant",
#     report_to="wandb"
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     max_seq_length= None,
#     dataset_text_field="chat_sample",
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing= False,
# )

# trainer.train()
# # Save the fine-tuned model
# trainer.model.save_pretrained(new_model)
# wandb.finish()
# model.config.use_cache = True
# model.eval()



# # def chatml_format(example):

# #     if len(example['system']) > 0:
# #         message = {"role": "system", "content": example['system']}
# #         system = tokenizer.apply_chat_template([message], tokenize=False)
# #     else:
# #         system = ""

# #     message = {"role": "user", "content": example['question']}
# #     prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
# #      # Format chosen answer
# #     chosen = example['chosen'] + "<|im_end|>\n"

# #     # Format rejected answer
# #     rejected = example['rejected'] + "<|im_end|>\n"

# #     return {
# #         "prompt": system + prompt,
# #         "chosen": chosen,
# #         "rejected": rejected,
# #     }