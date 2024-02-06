from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# classifer = pipeline("zero-shot-classification")
# res = classifer("This is a course about Python list comprehension",
#                 candidate_labels=["education", "politics", "business"],)
# print(res)

# print(pipeline('text-generation', 'mistralai/Mistral-7B-v0.1'))

model_name="mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False,revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt = "What is the capital of Australia?"
prompt_template=f'''{prompt}
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# classifer = pipeline("sentiment-analysis", model=model)
# res = classifer("I've been waiting for a HuggingFace course!")
# print(res)