import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Reference sentences
references = [
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

hypotheses = [
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

# Calculate BLEU score with smoothing
total_bleu_score = 0
smoother = SmoothingFunction()
for reference, hypothesis in zip(references, hypotheses):
    reference = [ref[0].split() for ref in reference]  # Split reference sentences into tokens
    hypothesis = hypothesis.split()  # Split hypothesis sentence into tokens
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoother.method7)
    print("BLEU score:", bleu_score)
    total_bleu_score += bleu_score

average_bleu_score = total_bleu_score / len(hypotheses)
print("Average BLEU score:", average_bleu_score)
