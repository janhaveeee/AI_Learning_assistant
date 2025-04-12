import numpy as np
import pandas as pd
import ast
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("student_learning_data_expanded.csv")

# Print column names to verify target column
print("Dataset Columns:", df.columns)

# Convert 'past_quiz_scores' from string to a numeric feature (average score)
df["past_quiz_scores"] = df["past_quiz_scores"].apply(
    lambda x: np.mean(ast.literal_eval(x)) if isinstance(x, str) else x
)

# Create 'difficulty_level' column based on quiz_score
df["difficulty_level"] = df["quiz_score"].apply(lambda x: 1 if x < 50 else 0)

# Encode categorical features
subject_encoder = LabelEncoder()
df["subject"] = subject_encoder.fit_transform(df["subject"])

study_time_encoder = LabelEncoder()
df["study_time_preference"] = study_time_encoder.fit_transform(df["study_time_preference"])

consistency_encoder = LabelEncoder()
df["study_consistency"] = consistency_encoder.fit_transform(df["study_consistency"])

revision_encoder = LabelEncoder()
df["Revision_Urgency"] = revision_encoder.fit_transform(df["Revision_Urgency"])

# Select relevant features
relevant_features = [
    "time_spent", "retry_attempts", "videos_watched", "articles_read",
    "quizzes_attempted", "interactive_exercises", "subject"
]

X = df[relevant_features]
y = df["difficulty_level"]  # Target: 0 = Strong, 1 = Weak

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Random Forest with class balancing
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)

# Evaluate Model
print(" Model: Random Forest (Balanced)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation for reliability check
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-validation Accuracy:", np.mean(cv_scores))

# Ensure the "models" directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save the model and scaler
joblib.dump(rf_model, os.path.join(model_dir, "random_forest_model_level.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print("✅ Model and scaler saved to 'models' directory.")
