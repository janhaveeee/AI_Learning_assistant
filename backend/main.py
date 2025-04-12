from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

from tutor import tutor_bp


app = Flask(__name__)
CORS(app)  

app.register_blueprint(tutor_bp)

# Define model paths
model_path = os.path.join("models", "proficiency_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
subject_encoder_path = os.path.join("models", "subject_encoder.pkl")

# ✅ New learning style model paths
learning_style_model_path = os.path.join("models", "learning_style_classifier.pkl")
learning_style_scaler_path = os.path.join("models", "learning_style_scaler.pkl")
subject_encoder_learning_style_path = os.path.join("models", "subject_encoder.pkl")
target_encoder_path = os.path.join("models", "target_encoder.pkl")

# Load proficiency model and encoders
if all(os.path.exists(p) for p in [model_path, scaler_path, subject_encoder_path]):
    log_reg = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    subject_encoder = joblib.load(subject_encoder_path)
else:
    log_reg, scaler, subject_encoder = None, None, None

# ✅ Load learning style model and scaler
required_learning_style_files = [
    learning_style_model_path, 
    learning_style_scaler_path,
    subject_encoder_learning_style_path,
    target_encoder_path
]

if all(os.path.exists(p) for p in required_learning_style_files):
    learning_style_model = joblib.load(learning_style_model_path)
    learning_style_scaler = joblib.load(learning_style_scaler_path)
    subject_encoder_learning_style = joblib.load(subject_encoder_learning_style_path)
    target_encoder = joblib.load(target_encoder_path)
else:
    learning_style_model, learning_style_scaler, subject_encoder_learning_style, target_encoder = None, None, None, None

# Features for proficiency model
FEATURES = [
    "time_spent", "retry_attempts", "videos_watched", "articles_read",
    "quizzes_attempted", "interactive_exercises", "subject"
]

# ✅ Features for learning style model (match the features used during training)
LEARNING_STYLE_FEATURES = [
    "time_spent", "retry_attempts", "videos_watched", "articles_read",
    "quizzes_attempted", "interactive_exercises", "subject"
]

# ✅ Default route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Learning Assistant API!"})

# ✅ Proficiency prediction route
# ✅ Proficiency prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if models are loaded
        if log_reg is None or scaler is None or subject_encoder is None:
            abort(500, description="Model or encoders not loaded properly.")

        data = request.get_json()
        df = pd.DataFrame([data])

        # Encode subject
        if "subject" in df.columns:
            try:
                df["subject"] = subject_encoder.transform([df["subject"].iloc[0]])[0]
            except ValueError as e:
                return jsonify({"error": f"Unknown subject: {df['subject'].iloc[0]}"}), 400

        else:
            return jsonify({"error": "Missing 'subject' in input data"}), 400

        # Scale numerical features
        scaled_features = scaler.transform(df[FEATURES])

        # Make prediction
        prediction = log_reg.predict(scaled_features)[0]
        probability = log_reg.predict_proba(scaled_features)[0].max()

        # ✅ Convert to native Python types
        return jsonify({
            "proficiency": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Learning style prediction route
@app.route("/predict-learning-style", methods=["POST"])
def predict_learning_style():
    try:
        # Check if models are loaded
        if any(x is None for x in [learning_style_model, learning_style_scaler, subject_encoder_learning_style, target_encoder]):
            abort(500, description="Learning style model or encoders not loaded properly.")

        data = request.get_json()
        
        # Create DataFrame with single row
        df = pd.DataFrame([data])
        
        # Check required features
        for feature in LEARNING_STYLE_FEATURES:
            if feature not in df.columns and feature != "subject":
                return jsonify({"error": f"Missing feature '{feature}' in input data"}), 400
        
        # Encode subject if present
        if "subject" in df.columns:
            try:
                df["subject"] = subject_encoder_learning_style.transform([df["subject"].iloc[0]])[0]
            except:
                return jsonify({"error": f"Unknown subject: {df['subject'].iloc[0]}"}), 400
        else:
            return jsonify({"error": "Missing 'subject' in input data"}), 400
            
        # Scale features
        scaled_features = learning_style_scaler.transform(df[LEARNING_STYLE_FEATURES])
        
        # Predict
        prediction = learning_style_model.predict(scaled_features)[0]
        probabilities = learning_style_model.predict_proba(scaled_features)[0]
        
        # Convert numeric prediction to learning style label
        learning_style = target_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary with learning style names
        class_labels = target_encoder.inverse_transform(range(len(probabilities)))
        probability_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        
        return jsonify({
            "learning_style": learning_style,
            "probabilities": probability_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Random Forest model (difficulty_level)
random_forest_model_path = os.path.join("models", "random_forest_model_level.pkl")
random_forest_scaler_path = os.path.join("models", "scaler.pkl")  # same scaler name as before!

if all(os.path.exists(p) for p in [random_forest_model_path, random_forest_scaler_path, subject_encoder_path]):
    random_forest_model = joblib.load(random_forest_model_path)
    random_forest_scaler = joblib.load(random_forest_scaler_path)
    subject_encoder = joblib.load(subject_encoder_path)  # already exists, safe to reload
else:
    random_forest_model, random_forest_scaler = None, None

@app.route("/predict_difficulty_level", methods=["POST"])
def predict_difficulty_level():
    if random_forest_model is None or random_forest_scaler is None:
        abort(500, description="Random Forest model or scaler not loaded properly.")

    data = request.get_json()

    try:
        # Prepare input DataFrame
        input_data = pd.DataFrame([data])

        # Encode 'subject'
        input_data['subject'] = subject_encoder.transform(input_data['subject'])

        # Scale the input features
        scaled_features = random_forest_scaler.transform(input_data[FEATURES])

        # Predict difficulty level
        prediction = random_forest_model.predict(scaled_features)

        # Prepare response
        response = {
            "difficulty_level": int(prediction[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Stress level model
# Corrected stress model loading
stress_model_path = os.path.join("models", "stress_regressor.pkl")

if os.path.exists(stress_model_path):
    stress_model = joblib.load(stress_model_path)
else:
    stress_model = None


# Features used during model training
STRESS_FEATURES = [
    "time_spent",
    "retry_attempts",
    "avg_past_quiz_score",
    "study_hours_per_week",
    "mistakes_made",
    "quizzes_attempted"
]


@app.route("/predict-stress", methods=["POST"])
def predict_stress_level():
    try:
        if stress_model is None:
            abort(500, description="Stress model not loaded.")

        data = request.get_json()
        df = pd.DataFrame([data])

        # Prepare input and predict
        input_data = df[STRESS_FEATURES]
        stress_prediction = stress_model.predict(input_data)[0]

        # Round to nearest integer category
        rounded_prediction = int(round(stress_prediction))
        stress_map = {0: "Low", 1: "Moderate", 2: "High"}
        stress_label = stress_map.get(rounded_prediction, "Unknown")

        return jsonify({
            "stress_level": stress_label,
            "numeric_value": float(stress_prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
