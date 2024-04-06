'''
This file was used for initial exploration and experimentation with an AI model.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer

# classifer = pipeline("zero-shot-classification")
# res = classifer("This is a course about Python list comprehension",
#                 candidate_labels=["education", "politics", "business"],)
# print(res)

# print(pipeline('text-generation', 'mistralai/Mistral-7B-v0.1'))

model_name="openai-community/gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda') #, device_map="auto", trust_remote_code=False,revision="main")
tokenizer = GPT2Tokenizer.from_pretrained(model_name) #, use_fast=True

prompt = "What is the capital of Australia?"
prompt_template=f'''{prompt}
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.5, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=20)
print(tokenizer.decode(output[0]))

# classifer = pipeline("sentiment-analysis", model=model)
# res = classifer("I've been waiting for a HuggingFace course!")
# print(res)