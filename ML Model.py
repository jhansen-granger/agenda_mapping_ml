import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

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

# 2. Feature Extraction
# Feature Extraction: Convert the text data into numerical features that the machine learning model can understand.
# You can use techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings such as Word2Vec
# or GloVe for this purpose.

# Initialize TfidfVectorizer with any custom parameters you may have used
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
X = tfidf_vectorizer.fit_transform(data['cleaned_proposal_longtext'])
y = data['Research_Issue_Code']

# Inspect Vocabulary
print("Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

# 2. Check Transformed Features
# Convert the transformed features to an array for easier inspection
X_array = X.toarray()

# Print the shape of the transformed features matrix
print("Shape of transformed features:", X_array.shape)

# Print the first few rows of the transformed features matrix
print("Transformed features (first few rows):")
print(X_array[:5])  # Print only the first 5 rows for brevity

#PICKUP HERE NEXT TIME - DON'T THINK VECTORIZING IS ACCURATE. Dictionary actually okay, but first few rows values seem
#wrong

# 3. Model Selection and Training
# Model Selection and Training: Choose a suitable machine learning algorithm such as Multinomial Naive Bayes,
# Support Vector Machines (SVM), or even deep learning models like Recurrent Neural Networks (RNNs) or Transformers.
# Train the model on your prepared dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Evaluation
# Evaluation: Evaluate the performance of your model using appropriate evaluation metrics such as accuracy,
# precision, recall, and F1-score. You can use techniques like cross-validation to ensure robustness of your model.
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Prediction
# Prediction and Deployment: Once satisfied with the model's performance, you can use it to predict the "Issue Codes"
# and consistent naming conventions for new agenda items.

# Load the Excel file with raw proposal text
excel_file = r"C:\Users\jhans\PycharmProjects\Agenda Mapping ML\Data\Test_Proposal_Data_1.xlsx"  # Provide the path to your Excel file

try:
    new_proposals_df = pd.read_excel(excel_file)
    # Loop through each row in the Excel file
    for index, row in new_proposals_df.iterrows():
        # Extract proposal text from the current row
        proposal_text = row['proposal_text']

        # Vectorize the proposal text using the same TfidfVectorizer used during training
        X_new = tfidf_vectorizer.transform([proposal_text])

        # Perform model prediction
        predicted_issue_code = model.predict(X_new)[0]

        # Update the "issue_codes" column with the predicted value
        new_proposals_df.at[index, 'issue_code'] = predicted_issue_code

    # Save the updated DataFrame back to the Excel file
    new_proposals_df.to_excel("new_proposals_with_predictions.xlsx", index=False)
except Exception as e:
    print(f"Error reading Excel file at {proposal_text}", e)

#Next steps
#Provide more sample proposal names to test with through an excel file
#Ideally have it write the issue code into that file for review
#Train it with a larger data set
#Figure out how to save the model for more efficient reuse

# Convert original names into consistent naming convention

# Deployment: Save the model for future use
# Example: joblib.dump(model, 'agenda_classifier_model.joblib')
