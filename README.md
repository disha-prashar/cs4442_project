# Dialogue Generation for NPCs with AI

This project was built for the CS4470 Capstone course at UWO by four senior undergraduate students. The primary focus of this project is to develop a system that automates NPC dialogue generation using AI and NLP techniques integrated within the Unity game engine.

This repository contains code that was used to fine-tune the Zephyr 7B as well as the Python server that was used to facilitate communication between the LLM and Unity engine.

## Fine-Tuning Zephyr 7B Model (Fine-tuned from Mistral 7B) 
The following steps were used to fine-tune Zephyr 7B
1. Use save_model_local.py to download and save the base Zephyr 7B model locally.
2. Use fine_tuning_data/convert_txt_to_json.py to preprocess the data found [here](https://jakub.thebias.nl/GPT2_WOWHead_dataset.txt) and convert it to a JSON file.
3. Upload the JSON file and create a dataset on the HuggingFace cloud [here](https://huggingface.co/datasets/dprashar/npc_dialogue_rpg_quests) which can be used to fine-tune Zephyr.
4. Use train_zephyr.py to fine-tune the downloaded Zephyr model on the dataset uploaded to HuggingFace.
5. Test the model's accuracy using test_trained_model.py

### Directories and Their Contents
archive: files in this directory were used for testing and experimentation

downloaded_model (only on computer in directory "C:\Users\NomadXR\Desktop\mistral_4470"): contains the Mistral 7B model downloaded straight from their [Github](https://github.com/mistralai)

fine_tuning_data: contains files that were used to create the dataset to fine-tune the model

llama.cpp (only on computer in directory "C:\Users\NomadXR\Desktop\mistral_4470"): external tool cloned from [GitHub](https://github.com/ggerganov/llama.cpp/discussions/2948) used to convert the final LLM to a gguf file to be used with Unity engine

All testing and fine-tuning should be done in the virtual environment. To activate the virtual environment in the terminal, use the following command in the directory "C:\Users\NomadXR\Desktop\mistral_4470"
```
.env/Scripts/activate
```
To deactivate:
```
deactivate
```

### Install dependencies
```
pip install transformers datasets
pip install -U optimum
```

## Python Server
