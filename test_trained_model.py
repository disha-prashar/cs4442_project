from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Test Model
local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/downloaded_model/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(local_model_directory) #.to('cuda') #model_name, device_map="auto", trust_remote_code=False,revision="main")
tokenizer = AutoTokenizer.from_pretrained(local_model_directory, use_fast=True)

user_content = '''
You are a creative writer working on a comedy novel.

In the style of Oscar Wilde, write the first paragraph of
a story where a guy accidentally meets the love of his life.
'''
messages = [{("user", "content"): user_content}]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

# prompt = "What is Scott McCall's biography?" # "Can you tell me about your transformation?"
# prompt_template=f'''{prompt}'''

# print("\n\n*** Generate:")

# input_ids = tokenizer(prompt_template, return_tensors='pt').to('cuda') #.input_ids.cuda()
# print(input_ids)
# output = model.generate(inputs=input_ids, max_new_tokens=512, do_sample=True) #temperature=0.7, 
# print(output)
# decoded = tokenizer.batch_decode(output)
# print(decoded[0])
# print(tokenizer.decode(output[0]))