import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from JSON file
df = pd.read_json('./cleaning_data/output.json')

# Tokenize and count words in each text field
word_counts_title = df['Title'].str.split().apply(len)
word_counts_objective = df['Objective'].str.split().apply(len)
word_counts_description = df['Text'].str.split().apply(len)

# Plotting
plt.figure(figsize=(10, 6))
plt.hist([word_counts_title, word_counts_objective, word_counts_description], bins=20, label=['Title', 'Objective', 'Description'])
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Word Frequency Distribution')
plt.legend()
plt.show()
