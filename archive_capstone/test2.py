'''
This file was used for initial exploration and experimentation with an AI model.
'''

from transformers import AutoTokenizer, MistralForCausalLM

model_name="mistralai/Mistral-7B-v0.1"
model = MistralForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Act like a potion seller in the medieval times in a video game."
prompt_template=f'''{prompt}'''
inputs = tokenizer(prompt_template, return_tensors="pt")

print("\n\n*** Generate:")

generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))
