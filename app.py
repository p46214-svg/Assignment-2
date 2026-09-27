import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained models ---
# Ensure the 'models' directory exists and models are saved there
MODELS_DIR = "models"

try:
    logistic_model = joblib.load(os.path.join(MODELS_DIR, "logistic_churn_model.pkl"))
    linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_monthly_charge_model.pkl"))
    print("Models loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model files not found in '{MODELS_DIR}'. Please ensure they are saved.")
    logistic_model = None
    linear_model = None

# --- Helper function for risk classification ---
def classify_risk(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"

# --- Prediction function (adapted from notebook) ---
def predict_customer_for_api(customer_data_dict):
    if logistic_model is None or linear_model is None:
        return {"error": "Models not loaded. Cannot make predictions."}

    # Convert input dictionary to DataFrame (single row)
    customer_df = pd.DataFrame([customer_data_dict])

    # Ensure columns match training data order and types if necessary
    # For simplicity, we assume incoming data has the same structure as X_logistic/X_linear
    # In a production setting, robust validation and alignment would be crucial.
    
    # CHURN PREDICTION
    churn_probability = logistic_model.predict_proba(customer_df)[0][1]
    churn_prediction = logistic_model.predict(customer_df)[0]
    risk = classify_risk(churn_probability)

    # MONTHLY CHARGE PREDICTION
    # Drop columns not used by the linear model (customerID, TotalCharges, Churn if present)
    linear_input = customer_df.drop(columns=["TotalCharges"], errors="ignore").copy()
    # The original notebook's linear model input also explicitly dropped 'Churn'.
    linear_input = linear_input.drop(columns=["Churn"], errors="ignore")
    
    # Make sure all columns expected by the linear preprocessor are present
    # This part might need more robust handling if input columns vary
    # For this example, we assume customer_df has all necessary features.

    monthly_charge_prediction = linear_model.predict(linear_input)[0]

    return {
        "Churn Probability (%)": round(churn_probability * 100, 2),
        "Predicted Churn": "Yes" if churn_prediction == 1 else "No",
        "Risk Category": risk,
        "Predicted Monthly Charges": round(monthly_charge_prediction, 2)
    }

# --- Flask API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) # Get data posted as JSON
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # The `predict_customer_for_api` expects a dictionary (representing a single customer)
    # If multiple customers are sent, this would need to be adapted.
    prediction_result = predict_customer_for_api(data)

    if "error" in prediction_result:
        return jsonify(prediction_result), 500
    
    return jsonify(prediction_result)


@app.route('/')
def home():
    return "Welcome to the Customer Prediction API! Post JSON data to /predict for predictions."


# This block allows running the app directly from this script
# It's typically used during development. For deployment, you might use a production-ready WSGI server like Gunicorn.
if __name__ == '__main__':
    # Using 0.0.0.0 makes the server accessible from outside the container (if deployed)
    app.run(host='0.0.0.0', port=5000, debug=True)
