import nltk
from nltk import pos_tag
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Step 1: Read the Moby Dick file from the Gutenberg dataset
moby_dick_text = gutenberg.raw('melville-moby_dick.txt')

# Step 2: Tokenization
tokens = word_tokenize(moby_dick_text)

# Step 3: Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Step 4: POS tagging
pos_tagged_tokens = pos_tag(filtered_tokens)

# Step 5: POS frequency
pos_freq = Counter(tag for word, tag in pos_tagged_tokens)
common_pos = pos_freq.most_common(5)

print("\n5 Most Common Parts of Speech and Their Frequency:")
for pos, freq in common_pos:
    print(f"{pos}: {freq}")

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens[:20]]

print("\nLemmatized Tokens:")
print(lemmatized_tokens)

# Step 7: Plotting frequency distribution
plt.figure(figsize=(10, 5))
plt.bar(*zip(*pos_freq.most_common(10)))
plt.title('Frequency Distribution of POS')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency')
plt.show()

# Optional: Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiments = sia.polarity_scores(moby_dick_text)

print("\nSentiment Analysis:")
print(sentiments)

if sentiments['compound'] > 0.05:
    print("The overall text sentiment is positive.")
else:
    print("The overall text sentiment is negative.")