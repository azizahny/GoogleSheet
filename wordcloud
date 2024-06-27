import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
from collections import defaultdict

# Define the scope
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Add your service account key file, you can adjust your location
creds = ServiceAccountCredentials.from_json_keyfile_name('credential_key.json', scope)

# Authorize the client
client = gspread.authorize(creds)

# Open the Google Sheet by its name
spreadsheet = client.open("Playstore Rating 2024")

# Select the targetted sheet
sheet_name = "Sheet1"
sheet = spreadsheet.worksheet(sheet_name)

# Get all values in the sheet
data = sheet.get_all_values()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data[1:], columns=data[0]) #Assuming first row is header

# Replace empty strings with NaN (if necessary)
df.replace('', pd.NA, inplace=True)

# Check columns and description of each columns
print(df.info())

# Set up stopwords in Bahasa Indonesia and English, then join both stopwords
indonesian_stopwords = stopwords.words('indonesian')
english_stopwords = stopwords.words('english')
all_stopwords = stopwords_indonesian.union(stopwords_english)

# If you want to see each stopwords
print(english_stopwords)
print(indonesian_stopwords)
print(all_stopwords)

# Delete null in df[‘Review Text’]
drop_null_review = df['Review Text'].dropna(inplace=True)

# Join each review text
text = " ".join(df['Review Text'].astype(str))

# Function to clean text from stopwords
def remove_stopwords(text):
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if word not in all_stopwords]
    return ' '.join(filtered_words)

# Function to clean text by removing non-alphabetic characters
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', str(text))

# Apply text cleaning
clean_text = df['Review Text'].apply(clean_text)

# Apply remove_stopwords function to 'Review Text' column
df['Cleaned Review Text'] = clean_text.apply(remove_stopwords)

# Compare before and after dataframe
print(df[['Review Text', 'Cleaned Review Text']])

# Convert Rating column to numeric (assuming it's stored as text)
df['Star Rating'] = pd.to_numeric(df['Star Rating'])

# Separate data into good (4-5) and bad (1-3) ratings
good_reviews = ' '.join(df[df['Star Rating'] >= 4]['Cleaned Review Text'])
bad_reviews = ' '.join(df[df['Star Rating'] <=3]['Cleaned Review Text'])

good_tokens = word_tokenize(good_reviews)
bad_tokens = word_tokenize(bad_reviews)

# Calculate word frequencies
good_freq = FreqDist(good_tokens)
bad_freq = FreqDist(bad_tokens)

# Display the most common words
print("Good Reviews - Most Common Words:")
print(good_freq.most_common())
print("\nBad Reviews - Most Common Words:")
print(bad_freq.most_common())

# Since there are so many words appear, we can categories them
categories = {
    'satisfaction': ['best', 'good', 'super', 'understand', 'wow', 'helpful', 'ok', 'easy', 'happy', 'bagus', 'membantu', 'mudah'],
    'dissatisfaction': ['bad', 'worst', 'ugly', 'jelek', 'buruk'],
    'payment': ['free', 'payment', 'pay', 'subscription', 'money', 'expensive', 'cheap', 'financial', 'cost', 'transfer', 'bayar', 'gratis', 'langganan', 'uang', 'mahal', 'murah', 'financial', 'finance', 'biaya'],
    'trial': ['test', 'try', 'trial', 'testing', 'test', 'uji', 'coba'],
    'application': ['feature', 'fitur', 'application', 'aplikasi', 'aplikasinya', 'platform', 'interface', 'app', 'perbarui', 'memperbarui', 'update', 'mengupdate'],
    'package': ['package', 'paket'],
    'service': ['service', 'customer', 'layanan', 'pelanggan', 'admin', 'adminnya'],
    'exclude-word': ['kak','ka', 'sis', 'bro', 'gk', 'banget', 'bgt', 'yg', 'yang', 'ngk', 'engga', 'tdk', 'gak', 'online']
}

# Doing tokenization from the categories
def categorize_word(tokens):
    token_categories = []
    for token in tokens:
        assigned_category = None
        for category, keywords in categories.items():
            if token in keywords:
                assigned_category = category
                break
        token_categories.append((token, assigned_category))
    return token_categories

# Count how many categories appears based on our tokens before
def count_categories(tokens):
    category_counts = defaultdict(int)
    for token in tokens:
        token_excluded = False
        for category, keywords in categories.items():
            if category == 'exclude-word' and token in keywords:
                token_excluded = True
                break
            elif token in keywords:
                category_counts[category] += 1
                break
    return category_counts

# Count and print good token and bad token categories
good_token_categories = categorize_word(good_tokens)
bad_token_categories = categorize_word(bad_tokens)

good_category_counts = count_categories(good_tokens)
bad_category_counts = count_categories(bad_tokens)

print("Good Token Category Counts:")
print(good_category_counts)
print("\nBad Token Category Counts:")
print(bad_category_counts)

# Set good and bad wordcloud
good_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(good_category_counts)
bad_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bad_category_counts)

# Plotting the word clouds
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(good_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Good Ratings (4-5)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(bad_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Bad Ratings (1-3)')
plt.axis('off')

plt.show()

# Calculate average rating
average_rating = df['Star Rating'].mean()
print(average_rating)

# Calculate frequency each of rating (1-5)
rating_counts = df['Star Rating'].value_counts().sort_index()
print(rating_counts)
rating_percentages = rating_counts / rating_counts.sum() * 100

#Plotting the bar chart
plt.figure(figsize=(6, 5))
bars = plt.bar(rating_counts.index.astype(str), rating_counts.values, color='skyblue')

# Adding values on top of bars
for bar, value in zip(bars, rating_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), value, ha='center', va='bottom')

plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Counts and Percentages')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Showing percentages on the right side
ax2 = plt.gca().twinx()
ax2.set_ylabel('Percentage')
ax2.set_ylim(0, 100)
ax2.set_yticks(range(0, 101, 10))

plt.show()

# Calculate rating without review
no_review_count = df['Review Text'].isnull().sum()

# Calculate rating with review
review_count = df['Review Text'].notnull().sum()

# Pie chart for review vs no review
plt.subplot(1, 2, 2)
labels = ['Review', 'No Review']
sizes = [review_count, no_review_count]
colors = ['skyblue', 'palevioletred']
explode = (0.1, 0)  # explode the 1st slice (Review)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Review vs No Review')

plt.tight_layout()
plt.show()

# Counting bad ratings (1-3) and good ratings (4-5) with reviews and no reviews
bad_ratings_reviews = df[(df['Star Rating'] <= 3) & (df['Review Text'].notnull())].shape[0]
bad_ratings_no_reviews = df[(df['Star Rating'] <= 3) & (df['Review Text'].isnull())].shape[0]
good_ratings_reviews = df[(df['Star Rating'] > 3) & (df['Review Text'].notnull())].shape[0]
good_ratings_no_reviews = df[(df['Star Rating'] > 3) & (df['Review Text'].isnull())].shape[0]

# Create DataFrame for plotting
plot_data = pd.DataFrame({
    'Reviews': [bad_ratings_reviews, good_ratings_reviews],
    'No Reviews': [bad_ratings_no_reviews, good_ratings_no_reviews]
}, index =['Bad Ratings (1-3)', 'Good Ratings (4-5)'])

# Plotting stacked bar chart
ax = plot_data.plot(kind='bar', stacked=False, figsize=(10, 6), color=['skyblue', 'palevioletred'])

# Adding counts on top of bars
for bar in ax.patches:
    ax.annotate(f'{bar.get_height()}', 
                (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                ha='center', va='center', color='white', fontsize=10, fontweight='bold')

plt.xlabel('Rating Categories')
plt.ylabel('Count')
plt.title('Bad vs Good Ratings with Reviews vs No Reviews')
plt.legend(title='Review Status')
plt.tight_layout()
plt.show()
