import json
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load preprocessed data from the file
with open('preprocessed_data.json', 'r') as f:
    data = json.load(f)

# Load the model
model_name = "HuggingFaceH4/zephyr-7b-beta"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to perform inference and return predictions
def perform_inference(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Get true labels and predicted labels
true_labels = [sample['label'] for sample in data]
predicted_labels = [perform_inference(sample['text']) for sample in data]

# Calculate accuracy as a baseline performance metric
accuracy = accuracy_score(true_labels, predicted_labels)
print("Baseline Accuracy:", accuracy)
