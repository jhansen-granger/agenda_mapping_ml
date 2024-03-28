import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# 1. Load and preprocess the dataset
# Data cleaning, tokenization, etc.
# Data Preparation: Prepare your dataset containing the shareholder meeting agenda items along with their corresponding
# "Issue Codes" and original names. You'll need to pre-process the data, clean it, and possibly tokenize the text.

# 1.1 - Load the data set
data = pd.read_csv(r"C:\Users\jhans\PycharmProjects\Agenda Mapping ML\Data\Historical_Agendas_Mapped_Mar27_5000_Clean.csv")

# 1.2 - Explore the data set
# Display the first few rows of the dataset
print(data.head())

# Get information about the dataset
print(data.info())

# Summary statistics of the dataset
print(data.describe())
# 1.3 - Data cleaning. Missing values, remove dupes, handle inconsistencies
# 1.4 - Text preprocessing. Tokenization, removing stopwords (should I?), lowercasing, Lemmatization or stemming

# Text preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stopwords and non-alphabetic tokens
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to the agenda item names
data['cleaned_proposal_longtext'] = data['PROPOSAL_LONGTEXT'].apply(preprocess_text)

# 1.5 - Final dataset
# Final dataset after data preparation
print(data.head())