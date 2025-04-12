import numpy as np
import pandas as pd
import ast
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("student_learning_data_expanded.csv")

# Print column names to verify
print("Dataset Columns:", df.columns)

# Process 'past_quiz_scores' to extract average
df["past_quiz_scores"] = df["past_quiz_scores"].apply(lambda x: np.mean(ast.literal_eval(x)) if isinstance(x, str) else x)
df["avg_past_quiz_score"] = df["past_quiz_scores"]
df.drop(columns=["past_quiz_scores"], inplace=True)

# Define 'stress_level' based on rule-based logic
def define_stress_level(row):
    if row['retry_attempts'] >= 4 and row['study_hours_per_week'] < 10:
        return 2  # High stress
    elif row['retry_attempts'] >= 2 and row['study_hours_per_week'] < 15:
        return 1  # Moderate stress
    else:
        return 0  # Low stress

df["stress_level"] = df.apply(define_stress_level, axis=1)

# Select features (without subject)
features = [
    "time_spent", "retry_attempts",
    "avg_past_quiz_score", "study_hours_per_week",
    "mistakes_made", "quizzes_attempted"
]
X = df[features]
y = df["stress_level"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train, y_train)

# Cross-validation (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gbr_model, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("\nCross-Validation RMSE scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Predict and evaluate
y_pred = gbr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel: Gradient Boosting Regressor")
print("Mean Absolute Error:", round(mae, 4))
print("Root Mean Squared Error:", rmse)

# Classification report after rounding predictions
y_pred_rounded = np.round(y_pred).astype(int)
print("\nRounded Classification Report:")
print(classification_report(y_test, y_pred_rounded))
print("Confusion Matrix (Rounded):\n", confusion_matrix(y_test, y_pred_rounded))

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(gbr_model, os.path.join(model_dir, "stress_regressor.pkl"))
print(f"Model saved in '{model_dir}'")

# Load for inference
gbr_model = joblib.load(os.path.join(model_dir, "stress_regressor.pkl"))

# Input for prediction
print("\nEnter student data for stress level prediction:")
time_spent = float(input("Time spent on learning (hours): "))
retry_attempts = int(input("Retry attempts: "))
avg_quiz_score = float(input("Average past quiz score: "))
study_hours = float(input("Study hours per week: "))
mistakes_made = int(input("Mistakes made: "))
quizzes_attempted = int(input("Quizzes attempted: "))

# Prepare input
input_df = pd.DataFrame([[time_spent, retry_attempts, avg_quiz_score, study_hours, mistakes_made, quizzes_attempted]],
                        columns=features)

# Predict stress level
regression_score = gbr_model.predict(input_df)[0]
stress_level_rounded = int(round(regression_score))
stress_level_rounded = max(0, min(2, stress_level_rounded))  # Clamp between 0 and 2
stress_map = {0: "Low", 1: "Moderate", 2: "High"}

print(f"\n📊 Predicted Stress Score (Regression Output): {round(regression_score, 2)}")
print(f"🧠 Interpreted Stress Level: {stress_map[stress_level_rounded]}")
print(f"🔁 (Rounded from raw score {round(regression_score, 2)} → {stress_level_rounded})")
