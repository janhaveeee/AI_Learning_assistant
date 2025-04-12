import numpy as np
import pandas as pd
import ast
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Create model directory
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv("student_learning_data.csv")

# Step 2: Process features
print("Dataset Columns:", df.columns)

# Convert 'past_quiz_scores' to numeric (average score)
df["past_quiz_scores"] = df["past_quiz_scores"].apply(lambda x: np.mean(ast.literal_eval(x)) if isinstance(x, str) else x)

# Encode categorical features
subject_encoder = LabelEncoder()
df["subject"] = subject_encoder.fit_transform(df["subject"])

study_time_encoder = LabelEncoder()
df["study_time_preference"] = study_time_encoder.fit_transform(df["study_time_preference"])

consistency_encoder = LabelEncoder()
df["study_consistency"] = df["study_consistency"].fillna("Unknown")  # handle missing if any
df["study_consistency"] = consistency_encoder.fit_transform(df["study_consistency"])

# Step 3: Create target variable (learning style)
def assign_learning_style(row):
    if row['videos_watched'] > row['articles_read'] and row['videos_watched'] > row['interactive_exercises']:
        return 'Visual'
    elif row['interactive_exercises'] > row['videos_watched'] and row['interactive_exercises'] > row['articles_read']:
        return 'Kinesthetic'
    else:
        return 'Auditory'

df["learning_style"] = df.apply(assign_learning_style, axis=1)

# Step 4: Select features and target (removed 'videos_preference')
relevant_features = [
    "time_spent", "retry_attempts", "videos_watched", "articles_read",
    "quizzes_attempted", "interactive_exercises", "subject"
]

X = df[relevant_features]
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["learning_style"])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Logistic Regression with class balancing
log_reg = LogisticRegression(class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)

# Step 8: Make predictions and evaluate
y_pred = log_reg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Save models and encoders
# Save model
model_path = os.path.join(model_dir, "learning_style_model.pkl")
joblib.dump(log_reg, model_path)
print(f"Model saved as '{model_path}'")

# Save scaler
scaler_path = os.path.join(model_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as '{scaler_path}'")

# Save encoders
subject_encoder_path = os.path.join(model_dir, "subject_encoder.pkl")
joblib.dump(subject_encoder, subject_encoder_path)
print(f"Subject Encoder saved as '{subject_encoder_path}'")

study_time_encoder_path = os.path.join(model_dir, "study_time_encoder.pkl")
joblib.dump(study_time_encoder, study_time_encoder_path)
print(f"Study Time Encoder saved as '{study_time_encoder_path}'")

consistency_encoder_path = os.path.join(model_dir, "consistency_encoder.pkl")
joblib.dump(consistency_encoder, consistency_encoder_path)
print(f"Consistency Encoder saved as '{consistency_encoder_path}'")

target_encoder_path = os.path.join(model_dir, "target_encoder.pkl")
joblib.dump(target_encoder, target_encoder_path)
print(f"Target Encoder saved as '{target_encoder_path}'")
