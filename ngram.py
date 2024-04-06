import nltk
from nltk.util import ngrams
from collections import Counter

# Reference and hypothesis sentences
reference_sentences = [
    "Welcome, weary traveler! Step right up and peruse my wares. Potions of all kinds await you.",
    "Greetings, adventurer! Looking for a potion to enhance your abilities? You've come to the right place.",
    "Ahoy there! Need a potion to cure what ails you? Look no further, for I am the finest potion seller in all the land.",
    "Need potions? I've got potions for sale.",
    "Hello, I'm a potion seller. Would you like to buy some potions?",
    "Potions! Get your potions here! Best potions in town!",
    "I am a potion seller. Come buy my potions, they're magical!",
    "Welcome! I'm a potion seller. Take a look at my potions.",
    "Potion seller here! Come and get your potions!",
    "Greetings! I have potions for sale. Interested?",
    "Potions available! Come see the potion seller!"
]

hypothesis_sentences = [
    "I am a computer program, incapable of selling potions or assuming a role.",
    "Acting like a potion seller is beyond my capabilities as an AI language model.",
    "Sorry, but I am not capable of acting or assuming roles like a potion seller.",
    "As an AI, I lack the physical form or consciousness to portray a potion seller.",
    "I'm just a machine learning model; pretending to be a potion seller isn't within my abilities.",
    "I'm not a real person, so I can't act like a potion seller.",
    "Being a potion seller requires human interaction and physical presence, which I cannot emulate as an AI.",
    "Welcome, weary traveler! Step right up and peruse my wares. Potions of all kinds await you.",
    "Greetings, adventurer! Looking for a potion to enhance your abilities? You've come to the right place.",
    "Ahoy there! Need a potion to cure what ails you? Look no further, for I am the finest potion seller in all the land."
]

# Tokenize reference and hypothesis sentences
reference_tokens = [sentence.split() for sentence in reference_sentences]
hypothesis_tokens = [sentence.split() for sentence in hypothesis_sentences]

# Generate n-grams
n = 2  # Example for bigrams, change to desired n
reference_ngrams = [ngrams(tokens, n) for tokens in reference_tokens]
hypothesis_ngrams = [ngrams(tokens, n) for tokens in hypothesis_tokens]

# Flatten n-grams lists
reference_ngrams_flat = [ng for ngrams_list in reference_ngrams for ng in ngrams_list]
hypothesis_ngrams_flat = [ng for ngrams_list in hypothesis_ngrams for ng in ngrams_list]

# Count n-gram frequencies
reference_ngram_counts = Counter(reference_ngrams_flat)
hypothesis_ngram_counts = Counter(hypothesis_ngrams_flat)

# Example analysis
print("Top 10 most common bigrams in reference sentences:")
print(reference_ngram_counts.most_common(10))

print("\nTop 10 most common bigrams in hypothesis sentences:")
print(hypothesis_ngram_counts.most_common(10))

# Plot results
import matplotlib.pyplot as plt

# Get top N most common n-grams
top_n = 10
top_reference_ngrams = reference_ngram_counts.most_common(top_n)
top_hypothesis_ngrams = hypothesis_ngram_counts.most_common(top_n)

# Extract n-grams and their frequencies
reference_ngrams, reference_counts = zip(*top_reference_ngrams)
hypothesis_ngrams, hypothesis_counts = zip(*top_hypothesis_ngrams)

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.barh(range(len(reference_ngrams)), reference_counts, color='blue')
plt.yticks(range(len(reference_ngrams)), reference_ngrams)
plt.xlabel('Frequency')
plt.title('Top {} Most Common Bigrams in Reference Sentences'.format(top_n))
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(range(len(hypothesis_ngrams)), hypothesis_counts, color='orange')
plt.yticks(range(len(hypothesis_ngrams)), hypothesis_ngrams)
plt.xlabel('Frequency')
plt.title('Top {} Most Common Bigrams in Hypothesis Sentences'.format(top_n))
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()