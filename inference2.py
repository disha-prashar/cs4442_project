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

# Function to visualize inference results and save them to a file
def visualize_results(data, output_file):
    correct_predictions = 0
    incorrect_predictions = 0
    with open(output_file, 'w') as outfile:
        for sample, true_label in zip(data, true_labels):
            text = ' '.join([sample['Title'], sample['Objective'], sample['Text']])  # Combine text features
            predicted_label = assign_pseudo_labels(text)
            if predicted_label == true_label:
                correct_predictions += 1
            else:
                incorrect_predictions += 1
                result = f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {predicted_label}\n\n"
                outfile.write(result)
    print("Results saved to", output_file)
    print("Correct Predictions:", correct_predictions)
    print("Incorrect Predictions:", incorrect_predictions)

# Specify the output file path
output_file = "inference_results.txt"

# Visualize inference results for the data subset and save them to the output file
visualize_results(data_subset, output_file)
