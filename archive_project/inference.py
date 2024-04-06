'''
PERFORM INFERENCE ON FINE-TUNED MODEL
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle

# Load preprocessed data from the file
with open('preprocessed_data.pkl', 'rb') as f:
    df, tfidf_matrix = pickle.load(f)

# Load the model and tokenizer
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input tensors and convert input tensor to torch.LongTensor
input_data_tensors = torch.tensor(tfidf_matrix.toarray()).to(torch.long)
print("got tensors")
# Perform inference
with torch.no_grad():
    outputs = model(input_data_tensors)
print("got outputs")
# Get predictions
predictions = torch.argmax(outputs.logits, dim=1)
print("got predictions")

'''
ANALYZE RESULTS
'''
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming true_labels contains the ground truth labels for the data
true_labels = torch.randint(0, 2, (100,))  # Example ground truth labels
# true_labels = [0, 1, 1, 0, 1, 0]  # Example ground truth labels

# 1. Accuracy Evaluation
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

# 2. Error Analysis
errors = [(true, pred) for true, pred in zip(true_labels, predictions) if true != pred]
error_rate = len(errors) / len(true_labels)
print("Error Rate:", error_rate)
print("Errors:", errors)

# 3. Confusion Matrix Visualization
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()