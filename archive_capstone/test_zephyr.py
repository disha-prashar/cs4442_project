'''
This program was used to test the accuracy of the fine-tuned Zephyr model.
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_text(prompt, model, tokenizer, conversation_history, max_length=50):
    input_ids = tokenizer.encode(prompt + conversation_history, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.9)

    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    conversation_history = ""

    while True:
        prompt = input("You: ")
        conversation_history += " " + prompt
        generated_text = generate_text(prompt, model, tokenizer, conversation_history)
        print("Bot:", generated_text)

'''
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from collections import deque
import torch

def chat_with_model(model_name, max_history=10):
    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model for conversational AI
    conversational_pipeline = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

    # Define conversation history
    conversation_history = []

    # Define conversation loop
    while True:
        # Get user input
        user_input = input("You: ")

        # Add user input to conversation history
        conversation_history.append(user_input)

        # Trim conversation history if it exceeds max_history
        if len(conversation_history) > max_history:
            conversation_history = conversation_history[-max_history:]

        # Generate response
        # input_text = tokenizer.eos_token.join(conversation_history[-max_history:])
        # bot_response = conversational_pipeline(tokenizer.pad(conversation_history, padding=True, max_length=max_history, return_tensors="pt"), max_length=50, do_sample=True)
        bot_response = conversational_pipeline(" ".join(conversation_history))

        # Decode and print model response
        print("Bot:", bot_response[0]['generated_text'].split("\n")[-1])

        # Add model response to conversation history
        conversation_history.append(bot_response[0]['generated_text'].split("\n")[-1])

        # Check for exit command
        if user_input.lower() == 'exit':
            break

# Example usage
if __name__ == "__main__":
    model_name = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"
    chat_with_model(model_name)


def chat_with_model(model_name):
    # Load pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    conversation_history = []

    # Define conversation loop
    while True:
        # Get user input
        user_input = input("You: ")

        # Check for exit command
        if user_input.lower() == 'exit':
            break

        # Concatenate conversation history and user input
        input_text = ' '.join(conversation_history + [user_input])

        # Encode user input and generate response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        response_ids = model.generate(input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_length=100, pad_token_id=tokenizer.eos_token_id)

        # Decode and print model response
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print("Bot:", bot_response) 

        # Update conversation history
        conversation_history.append(user_input)
        conversation_history.append(bot_response)

if __name__ == "__main__":
    model_name = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"
    chat_with_model(model_name)
'''
# local_model_directory = "C:/Users/NomadXR/Desktop/mistral_4470/zephyr_model"

# pipe = pipeline("text-generation", model=local_model_directory, torch_dtype=torch.bfloat16, device_map="auto")

# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])


# SAVE MODEL LOCALLY
# model_name="HuggingFaceH4/zephyr-7b-alpha"
# model = AutoModelForCausalLM.from_pretrained(model_name)

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, src_lang="en")
# tokenizer.add_eos_token = True
# tokenizer.add_bos_token, tokenizer.add_eos_token
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"

# model.save_pretrained(local_model_directory)
# tokenizer.save_pretrained(local_model_directory)