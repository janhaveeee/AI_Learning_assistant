import numpy as np
import pandas as pd
import ast
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("student_learning_data_expanded.csv")

# Print column names to verify target column
print("Dataset Columns:", df.columns)

# Convert 'past_quiz_scores' from string to a numeric feature (average score)
df["past_quiz_scores"] = df["past_quiz_scores"].apply(lambda x: np.mean(ast.literal_eval(x)) if isinstance(x, str) else x)

#  Create 'difficulty_level' column based on quiz_score
df["difficulty_level"] = df["quiz_score"].apply(lambda x: 1 if x < 50 else 0)

# Encode categorical features
subject_encoder = LabelEncoder()
df["subject"] = subject_encoder.fit_transform(df["subject"])

study_time_encoder = LabelEncoder()
df["study_time_preference"] = study_time_encoder.fit_transform(df["study_time_preference"])

consistency_encoder = LabelEncoder()
df["study_consistency"] = consistency_encoder.fit_transform(df["study_consistency"])

revision_encoder = LabelEncoder()
df["Revision_Urgency"] = consistency_encoder.fit_transform(df["Revision_Urgency"])


#  Select only relevant features
relevant_features = [
    "time_spent", "retry_attempts", "videos_watched", "articles_read",
    "quizzes_attempted", "interactive_exercises", "subject"
]

X = df[relevant_features]  # Only use relevant features
y = df["difficulty_level"]  # Target (0 = Strong, 1 = Weak)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression with Class Balancing
log_reg = LogisticRegression(class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)

# Make Predictions
y_pred = log_reg.predict(X_test)

#  Evaluate Model
print(" Model: Logistic Regression (Balanced)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Ensure the "models" directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save trained model
model_path = os.path.join(model_dir, "proficiency_model.pkl")
joblib.dump(log_reg, model_path)
print(f" Model saved as '{model_path}'")

# Save other necessary encoders and scaler
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(subject_encoder, os.path.join(model_dir, "subject_encoder.pkl"))


# Load trained model and encoders
log_reg = joblib.load(os.path.join(model_dir, "proficiency_model.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
subject_encoder = joblib.load(os.path.join(model_dir, "subject_encoder.pkl"))

# Get the valid subject choices
valid_subjects = list(subject_encoder.classes_)

# Take user input
print("\nEnter test data for prediction:")
time_spent = float(input("Enter time spent on learning (hours): "))
retry_attempts = int(input("Enter number of retry attempts: "))
videos_watched = int(input("Enter number of videos watched: "))
articles_read = int(input("Enter number of articles read: "))
quizzes_attempted = int(input("Enter number of quizzes attempted: "))
interactive_exercises = int(input("Enter number of interactive exercises: "))

# Validate subject input
while True:
    subject = input(f"Enter subject ({', '.join(valid_subjects)}): ").strip()
    if subject in valid_subjects:
        break
    print(f"Subject '{subject}' not recognized. Choose from: {valid_subjects}")

# Encode categorical values
subject_encoded = subject_encoder.transform([subject])[0]

# Create DataFrame with the same feature names as in training
input_df = pd.DataFrame([[time_spent, retry_attempts, videos_watched, articles_read, 
                          quizzes_attempted, interactive_exercises, subject_encoded]], 
                        columns=relevant_features)

# Scale the input properly
input_scaled = scaler.transform(input_df)

# Predict difficulty level
prediction = log_reg.predict(input_scaled)
difficulty_label = "Weak" if prediction[0] == 1 else "Strong"

# Show result
print(f"\nPredicted Difficulty Level: {difficulty_label}")
