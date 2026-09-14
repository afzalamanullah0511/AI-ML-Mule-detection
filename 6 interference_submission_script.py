import pandas as pd
import xgboost as xgb

print("Loading the unseen evaluation dataset...")
# Replace with the actual file name provided by the hackathon organizers
test_df = pd.read_csv('evaluation_test_data.csv') 

# Keep track of the account identifier column if the organizers require it in submission
# (e.g., 'Account_ID' or an index column). Let's assume there's an 'ID' or index.
if 'ID' in test_df.columns:
    submission_ids = test_df['ID']
else:
    submission_ids = test_df.index

print("Applying identical behavioral feature matching...")
# 1. Strip the exact same database artifact columns we excluded during training
drop_cols = [col for col in test_df.columns if 'Unnamed' in col or 'F2230' in col]
for i in range(3900, 3925): # Drops up to F3924 if present
    col_str = f'F{i}'
    if col_str in test_df.columns:
        drop_cols.append(col_str)

# Ensure our feature matrices match precisely
X_eval = test_df.drop(columns=drop_cols, errors='ignore')

print("Initializing shell architecture...")
# Initialize an empty XGBoost Classifier structure
model = xgb.XGBClassifier()

print("Injecting the trained brain file...")
# Load the saved state directly from your JSON file
model.load_model('final_pure_behavioral_mule_hunter.json')

print("Generating target predictions...")
# 1. Generate hard binary labels (0 = Legitimate, 1 = Mule Account)
predictions = model.predict(X_eval)

# 2. Generate probability scores (often used for ROC-AUC grading)
probabilities = model.predict_proba(X_eval)[:, 1]

print("Compiling official submission dataframe...")
# Construct the leaderboard file as requested by IIT Hyderabad guidelines
submission_df = pd.DataFrame({
    'Account_Identifier': submission_ids,
    'Predicted_Class': predictions,
    'Mule_Probability_Score': probabilities
})

# Export to a clean CSV format
submission_output_path = 'final_leaderboard_submission.csv'
submission_df.to_csv(submission_output_path, index=False)

print(f"\n=============================================================")
print(f" SUCCESS: '{submission_output_path}' has been successfully created!")
print(f" Total rows processed: {len(submission_df)}")
print("=============================================================")
print("You are ready to upload this file directly to the portal evaluation dashboard.")