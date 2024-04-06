def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
    B_INST, E_INST = "[INST]", "[/INST]"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

stream("Explain large language models")

# Clear the memory footprint
del model, trainer
torch.cuda.empty_cache()



# # Format dataset
# dataset = dataset.map(
#     chatml_format,
#     #remove_columns=original_columns
# )



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