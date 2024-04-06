import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import random

# Load JSON data from file
with open('./cleaning_data/output.json', 'r') as f:
    data = json.load(f)

# Load the model and tokenizer
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to perform sentiment analysis and assign pseudo-labels
def assign_pseudo_labels(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return "positive" if predicted_class == 1 else "negative"

# Perform exploratory analysis and assign pseudo-labels
true_labels = []
for sample in data:
    text = ' '.join([sample['Title'], sample['Objective'], sample['Text']])  # Combine text features
    true_label = assign_pseudo_labels(text)  # Perform sentiment analysis and assign pseudo-label
    true_labels.append(true_label)

# Randomly sample a subset of the data for visualization
data_subset = random.sample(data, min(100, len(data)))

# Function to visualize inference results
def visualize_results(data):
    correct_predictions = 0
    incorrect_predictions = 0
    for sample, true_label in zip(data, true_labels):
        text = ' '.join([sample['Title'], sample['Objective'], sample['Text']])  # Combine text features
        predicted_label = assign_pseudo_labels(text)
        if predicted_label == true_label:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
            print("Text:", text)
            print("True Label:", true_label)
            print("Predicted Label:", predicted_label)
            print()
    print("Correct Predictions:", correct_predictions)
    print("Incorrect Predictions:", incorrect_predictions)

# Visualize inference results for the data subset
visualize_results(data_subset)
