import pandas as pd
import joblib
from preprocessing import preprocess_text  # Ensure this matches the training script

# Load the trained model and vectorizer
model = joblib.load('agenda_classifier_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load new proposals data
new_proposals_df = pd.read_excel(r"C:\Users\jhans\PycharmProjects\Agenda_Mapping_ML\Data\Test_Proposal_Data_1.xlsx")  # Provide the path to your Excel file

# Preprocess and predict
new_proposals_df['cleaned_text'] = new_proposals_df['proposal_text'].apply(preprocess_text)
X_new = tfidf_vectorizer.transform(new_proposals_df['cleaned_text'])
new_proposals_df['issue_code'] = model.predict(X_new)

# Save predictions to a new Excel file
new_proposals_df.to_excel("new_proposals_with_predictions2.xlsx", index=False)