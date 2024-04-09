# Dialogue Generation for NPCs with AI

This project is based off of a capstone project that was built by four undergraduate students for the CS4470 Capstone course at UWO. The primary focus of this project is to develop a system that automates NPC dialogue generation using AI and NLP techniques integrated within the Unity game engine.

This repository contains code that was used to fine-tune the Zephyr 7B.

### Install dependencies for visualizations
```
python -m pip install pandas matplotlib nltk wordcloud textblob seabor
python -m pip install -U scikit-learn
```

### Install dependencies for fine tuning
```
python -m pip install transformers datasets peft sentencepiece protobuf trl wandb
python -m pip install -U optimum
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Fine-Tuning Zephyr 7B Model (Fine-tuned from Mistral 7B) 
The following steps were used to fine-tune Zephyr 7B
1. Use save_model_local.py to download and save the base Zephyr 7B model locally.
2. Use fine_tuning_data/convert_txt_to_json.py to clean the data found [here](https://jakub.thebias.nl/GPT2_WOWHead_dataset.txt) and convert it to a JSON file.
3. Preprocess the JSON file and create two datasets (splits: train and test) stored in the dataset directory which can be used to fine-tune Zephyr.
4. Use train_zephyr.py to fine-tune the downloaded Zephyr model on the training dataset.
5. Test the model's accuracy using test_trained_model.py.

### Directories/Files and Their Contents
analysis_results: files in this directory were used to perform analysis for the results section and the outputs are available here.

archive_project: files in this directory were used for testing and experimentation and were not used for the final results.

cleaning_data: files in this directory were used to clean the data and to store the cleaned data.

dataset: stores the datasets resulting from preprocessing.

fine_tuning: contains files that were used to save, train, and test the model locally.

zephyr_model: contains the locally downloaded model.

preprocessing_eda.py: contains code used for preprocessing and exploratory data analysis (eda). the preprocessed data was stored in preprocessed_data.pkl.

All testing and fine-tuning should be done in the virtual environment. To activate the virtual environment in the terminal, use the following command
```
.env/Scripts/activate
```
To deactivate:
```
deactivate
```
