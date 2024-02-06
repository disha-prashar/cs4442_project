from transformers import AutoTokenizer, AutoModel

model_name="mistralai/Mistral-7B-v0.1"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470"
model.save_pretrained(local_model_directory)
tokenizer.save_pretrained(local_model_directory)

model.from_pretrained(local_model_directory)
tokenizer.from_pretrained(local_model_directory)