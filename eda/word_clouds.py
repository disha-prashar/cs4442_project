import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset from JSON file
df = pd.read_json('./cleaning_data/output.json')

# Concatenate text from all entries for each field
all_titles = ' '.join(df['Title'])
all_objectives = ' '.join(df['Objective'])
all_descriptions = ' '.join(df['Text'])

# Generate word clouds
wordcloud_titles = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
wordcloud_objectives = WordCloud(width=800, height=400, background_color='white').generate(all_objectives)
wordcloud_descriptions = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

# Plotting
plt.figure(figsize=(15, 10))
plt.subplot(131)
plt.imshow(wordcloud_titles, interpolation='bilinear')
plt.title('Word Cloud for Titles')
plt.axis('off')

plt.subplot(132)
plt.imshow(wordcloud_objectives, interpolation='bilinear')
plt.title('Word Cloud for Objectives')
plt.axis('off')

plt.subplot(133)
plt.imshow(wordcloud_descriptions, interpolation='bilinear')
plt.title('Word Cloud for Descriptions')
plt.axis('off')

plt.show()