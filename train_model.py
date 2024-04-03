import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from preprocessing import preprocess_text

# 1. Load and preprocess the dataset
# Data cleaning, tokenization, etc.
# Data Preparation: Prepare your dataset containing the shareholder meeting agenda items along with their corresponding
# "Issue Codes" and original names. You'll need to pre-process the data, clean it, and possibly tokenize the text.

# 1.1 - Load the data set
data = pd.read_csv(r"C:\Users\jhans\PycharmProjects\Agenda_Mapping_ML\Data\Historical_Agendas_Mapped_Mar27_5000_Clean.csv")

# # 1.2 - Explore the data set
# # Display the first few rows of the dataset
# print(data.head())
#
# # Get information about the dataset
# print(data.info())
#
# # Summary statistics of the dataset
# print(data.describe())
# # 1.3 - Data cleaning. Missing values, remove dupes, handle inconsistencies
# # 1.4 - Text preprocessing. Tokenization, removing stopwords (should I?), lowercasing, Lemmatization or stemming


# Apply preprocessing to the agenda item names
data['cleaned_proposal_longtext'] = data['PROPOSAL_LONGTEXT'].apply(preprocess_text)

# # 1.5 - Final dataset
# # Final dataset after data preparation
# print(data.head())

# 2. Feature Extraction
# Feature Extraction: Convert the text data into numerical features that the machine learning model can understand.
# You can use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings such as Word2Vec
# or GloVe for this purpose.

# Initialize TfidfVectorizer with any custom parameters you may have used
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
X = tfidf_vectorizer.fit_transform(data['cleaned_proposal_longtext'])
y = data['Research_Issue_Code']

# # Inspect Vocabulary
# print("Vocabulary:")
# print(tfidf_vectorizer.get_feature_names_out())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer for later use
joblib.dump(model, 'agenda_classifier_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
#
# # 2. Check Transformed Features
# # Convert the transformed features to an array for easier inspection
# X_array = X.toarray()
#
# # Print the shape of the transformed features matrix
# print("Shape of transformed features:", X_array.shape)
#
# # Print the first few rows of the transformed features matrix
# print("Transformed features (first few rows):")
# print(X_array[:5])  # Print only the first 5 rows for brevity


#Next steps
#Provide more sample proposal names to test with through an excel file
#Ideally have it write the issue code into that file for review
#Train it with a larger data set
#Figure out how to save the model for more efficient reuse

# Convert original names into consistent naming convention

# Deployment: Save the model for future use
# Example: joblib.dump(model, 'agenda_classifier_model.joblib')
