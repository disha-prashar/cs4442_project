# Dialogue Generation for NPCs with AI

This project is based off of a capstone project that was built by four undergraduate students for the CS4470 Capstone course at UWO. The primary focus of this project is to develop a system that automates NPC dialogue generation using AI and NLP techniques integrated within the Unity game engine.

This repository contains code that was used to fine-tune the Zephyr 7B.

### Install dependencies for visualizations
```
pip install pandas matplotlib nltk wordcloud textblob seaborn
```

### Install dependencies for fine tuning
```
pip install transformers datasets peft sentencepiece protobuf trl wandb
pip install -U optimum
```

## Fine-Tuning Zephyr 7B Model (Fine-tuned from Mistral 7B) 
The following steps were used to fine-tune Zephyr 7B
1. Use save_model_local.py to download and save the base Zephyr 7B model locally.
2. Use fine_tuning_data/convert_txt_to_json.py to preprocess the data found [here](https://jakub.thebias.nl/GPT2_WOWHead_dataset.txt) and convert it to a JSON file.
3. Upload the JSON file and create a dataset on the HuggingFace cloud [here](https://huggingface.co/datasets/dprashar/npc_dialogue_rpg_quests) which can be used to fine-tune Zephyr.
4. Use train_zephyr.py to fine-tune the downloaded Zephyr model on the dataset uploaded to HuggingFace.
5. Test the model's accuracy using test_trained_model.py.

### Directories and Their Contents
archive: files in this directory were used for testing and experimentation

fine_tuning_data: contains files that were used to create the dataset to fine-tune the model

All testing and fine-tuning should be done in the virtual environment. To activate the virtual environment in the terminal, use the following command
```
.env/Scripts/activate
```
To deactivate:
```
deactivate
```
