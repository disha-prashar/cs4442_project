'''
This program was used to test the accuracy of the fine-tuned Zephyr model.
'''
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def chat_with_model():
    local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"
    model = AutoModelForCausalLM.from_pretrained(local_model_directory) #.to('cuda') #(model_name, device_map="auto", trust_remote_code=False,revision="main")
    tokenizer = AutoTokenizer.from_pretrained(local_model_directory, use_fast=True, src_lang="en")
    
    conversation_history = ""

    while True:
        prompt = input("\nEnter prompt: ") #"What's the capital of Australia?"  "Who is Scott McCall?" "Act like a potion seller." "Can you tell me about your transformation?"
        # prompt_template=f'''{prompt}'''
        conversation_history += " " + prompt

        if prompt.lower() == "exit":
            break
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt") #.input_ids.cuda()

        print("\n*** Generate:\n")
        generate_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.9, do_sample=True) #top_p=0.95, top_k=40, max_new_tokens=512
        bot_response = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        conversation_history += " " + bot_response
        print(bot_response)

if __name__ == "__main__":
    chat_with_model()


# decoded = tokenizer.batch_decode(output)
# print(decoded[0])


# messages = [
#     {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
# ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))
# outputs = model.generate(tokenized_chat, max_new_tokens=128) 
# print(tokenizer.decode(outputs[0]))