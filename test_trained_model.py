from transformers import pipeline, AutoTokenizer, MistralForCausalLM

# Test Model
local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/local_model_2/"
model = MistralForCausalLM.from_pretrained(local_model_directory) #model_name, device_map="auto", trust_remote_code=False,revision="main")
tokenizer = AutoTokenizer.from_pretrained(local_model_directory, use_fast=True)

prompt = "What is Scott McCall's biography?" # "Can you tell me about your transformation?"
prompt_template=f'''{prompt}
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt') #.input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))