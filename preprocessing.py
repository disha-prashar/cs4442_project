'''
PREPROCESSING
'''

# Installing dependencies required for tokenization, stopword removal, lemmatization, and vectorization
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Dependencies for Preprocessing
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
df = pd.read_json('./cleaning_data/output.json')

# Data cleaning
# Remove unnecessary fields
df = df[['Title', 'Objective', 'Text']]

# Text cleaning
df['Title'] = df['Title'].str.lower()
df['Objective'] = df['Objective'].str.lower()
df['Text'] = df['Text'].str.lower()

# Remove punctuation
df['Title'] = df['Title'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['Objective'] = df['Objective'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['Text'] = df['Text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Tokenization
df['Title'] = df['Title'].apply(word_tokenize)
df['Objective'] = df['Objective'].apply(word_tokenize)
df['Text'] = df['Text'].apply(word_tokenize)

# Stopword removal
stop_words = set(stopwords.words('english'))
df['Title'] = df['Title'].apply(lambda x: [word for word in x if word not in stop_words])
df['Objective'] = df['Objective'].apply(lambda x: [word for word in x if word not in stop_words])
df['Text'] = df['Text'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['Title'] = df['Title'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['Objective'] = df['Objective'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['Text'] = df['Text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Text'].apply(' '.join))

# Save preprocessed data to a file
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((df, tfidf_matrix), f)

# Save dataset to upload to Hugging Face
# Import necessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import datasets

# Load the preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    df, _ = pickle.load(f)

# Convert pandas DataFrame to dictionary format
data_dict = df.to_dict(orient='list')

# Create a dataset object
my_dataset = datasets.Dataset.from_dict(data_dict)

# Splitting the dataset into train and test sets
train_size = int(len(my_dataset) * 0.8)
train_dataset = my_dataset.select(range(train_size))
test_dataset = my_dataset.select(range(train_size, len(my_dataset)))

# Save the train and test datasets
train_dataset.save_to_disk('./dataset/train_dataset')
test_dataset.save_to_disk('./dataset/test_dataset')

# # Upload the dataset to Hugging Face dataset repository
# datasets.DatasetDict({'train': my_dataset}).save_to_disk('./dataset/')

'''
EXPLORATORY DATA ANALYSIS (EDA)
'''

# # Dependencies for Exploratory Data Analysis (EDA)

# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from textblob import TextBlob
# import seaborn as sns

# # Word Frequency Analysis

# # Calculate word frequencies
# word_freq = {}
# for text in df['Text']:
#     for word in text:
#         if word in word_freq:
#             word_freq[word] += 1
#         else:
#             word_freq[word] = 1
# # Sort the word frequencies in descending order
# sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
# # Extract the top N most common words and their frequencies
# top_words = [word[0] for word in sorted_word_freq[:20]]
# top_freqs = [word[1] for word in sorted_word_freq[:20]]
# # Plot the most common words
# plt.figure(figsize=(10, 6))
# plt.bar(top_words, top_freqs)
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.title('Top 20 Most Common Words')
# plt.xticks(rotation=45)
# plt.savefig('word_frequency.png', bbox_inches='tight')
# plt.close()

# # Word Clouds for Title, Objective, and Text

# # Concatenate all words for each field
# title_text = ' '.join(df['Title'].apply(' '.join))
# objective_text = ' '.join(df['Objective'].apply(' '.join))
# text_text = ' '.join(df['Text'].apply(' '.join))
# # Plot word cloud for Titles
# plt.figure(figsize=(15, 5))
# wordcloud_title = WordCloud(width=800, height=400, background_color='white').generate(title_text)
# plt.imshow(wordcloud_title, interpolation='bilinear')
# plt.title('Word Cloud for Titles')
# plt.axis('off')
# plt.savefig('word_cloud_titles.png', bbox_inches='tight')
# plt.close()
# # Plot word cloud for Objectives
# plt.figure(figsize=(15, 5))
# wordcloud_objective = WordCloud(width=800, height=400, background_color='white').generate(objective_text)
# plt.imshow(wordcloud_objective, interpolation='bilinear')
# plt.title('Word Cloud for Objectives')
# plt.axis('off')
# plt.savefig('word_cloud_objectives.png', bbox_inches='tight')
# plt.close()
# # Plot word cloud for Descriptions
# plt.figure(figsize=(15, 5))
# wordcloud_text = WordCloud(width=800, height=400, background_color='white').generate(text_text)
# plt.imshow(wordcloud_text, interpolation='bilinear')
# plt.title('Word Cloud for Descriptions')
# plt.axis('off')
# plt.savefig('word_clouds_descriptions.png', bbox_inches='tight')
# plt.close()

# # Sentiment Analysis

# # Perform sentiment analysis
# df['Sentiment'] = df['Text'].apply(lambda x: TextBlob(' '.join(x)).sentiment.polarity)
# # Plot sentiment analysis
# plt.figure(figsize=(8, 6))
# sns.histplot(df['Sentiment'], bins=20, kde=True)
# plt.title('Sentiment Analysis of Text')
# plt.xlabel('Sentiment Polarity')
# plt.ylabel('Frequency')
# plt.savefig('sentiment_analysis.png', bbox_inches='tight')
