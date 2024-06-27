# GoogleSheet

This project showing how to work with Google Sheets data in Pandas, you can use the Google Sheets API to access the data programmatically

**1. Set Up Google Cloud Project and Enable the API**

* Create a project in the Google Cloud Console.
* Enable the Google Sheets API and the Google Drive API for your project. 
  Go to your project -> APIs & Services -> Search for Google Sheets API and Google Drive API then enable them.
* Create credentials for a service account and download the JSON key file.
  Tick the User Data option and then click done.

**2. Install Gspread**

Gspread is a Python library that allows you to interact with Google Sheets through the Google Sheets API. It provides a simple and convenient interface for reading, writing, and manipulating data in Google Sheets programmatically. This can be particularly useful for automating tasks that involve data stored in Google Sheets, such as data extraction, updating, and analysis. You'll need the `gspread` and `oauth2client` libraries, in addition to `pandas`.
`pip install gspread oauth2client pandas`

**3. Install Wordcloud**

WordCloud is a Python library used to create visualizations of text in Word Cloud form. A word cloud is a visual representation of the frequency with which words appear in text. This library can help in analyzing and visualizing the most common word patterns in text.
`pip install wordcloud`

**4. Install NLTK**
NLTK is a Python library that provides tools and resources for processing and analyzing text. SentimentIntensityAnalyzer is a class in NLTK that is used to analyze sentiment in text. SentimentIntensityAnalyzer calculates sentiment scores based on a rule-based sentiment analysis approach.
`pip install nltk` then download some resources such as `vader_lexicon`, `stopwords`, `punkt`.

* **Vader Lexicon**: The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon is a tool designed for sentiment analysis. It is specifically attuned to sentiments expressed in social media. The lexicon contains a list of lexical features (words) and their associated sentiment intensity scores. These scores range from -4 (most negative) to +4 (most positive), allowing the model to predict the sentiment of a given text. VADER can classify text as positive, negative, or neutral, and it can also provide a compound score that represents the overall sentiment of the text. It's particularly useful for analyzing sentiment in short texts like tweets, comments, and reviews.

* **Stopwords**: Stopwords are common words that usually do not carry significant meaning and are often filtered out in text preprocessing steps. Examples of stopwords include "is", "the", "and", "in", etc. The stopwords corpus in NLTK includes a list of these common words for various languages. Removing stopwords helps in focusing on the meaningful words in a text for tasks like text classification, topic modeling, and keyword extraction. Stopwords are used to clean the text by removing the noise (less meaningful words), which can improve the performance of NLP models.

* **Punkt**: Punkt is a sentence tokenizer that is included in NLTK. It helps in dividing a text into a list of sentences. Punkt uses unsupervised machine learning to build a model for abbreviation words, collocations, and words that start sentences. This process is critical for many NLP tasks where you need to analyze or process text at the sentence level, such as sentence segmentation, named entity recognition, and text summarization. Punkt can be used to split a text into sentences accurately, which is an essential step before further processing like parsing or extracting information.

  5. Import Matplotlib, Pandas
