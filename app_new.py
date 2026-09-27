import streamlit as st
import joblib
import pandas as pd
import os

# --- Load the trained models ---
MODELS_DIR = "models" # Ensure this directory exists relative to your Streamlit app

# Use st.cache_resource to load models only once
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load(os.path.join(MODELS_DIR, "logistic_churn_model.pkl"))
        linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_monthly_charge_model.pkl"))
        return logistic_model, linear_model
    except FileNotFoundError:
        st.error(f"Error: Model files not found in '{MODELS_DIR}'. Please ensure they are saved.")
        return None, None

logistic_model, linear_model = load_models()

# --- Helper function for risk classification ---
def classify_risk(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"

# --- Prediction function (adapted from notebook) ---
def predict_customer_for_streamlit(customer_data_dict):
    if logistic_model is None or linear_model is None:
        return {"error": "Models not loaded. Cannot make predictions."}

    customer_df = pd.DataFrame([customer_data_dict])

    # CHURN PREDICTION
    churn_probability = logistic_model.predict_proba(customer_df)[0][1]
    churn_prediction = logistic_model.predict(customer_df)[0]
    risk = classify_risk(churn_probability)

    # MONTHLY CHARGE PREDICTION
    linear_input = customer_df.drop(columns=["TotalCharges"], errors="ignore").copy()
    linear_input = linear_input.drop(columns=["Churn"], errors="ignore")
    monthly_charge_prediction = linear_model.predict(linear_input)[0]

    return {
        "Churn Probability (%)": round(churn_probability * 100, 2),
        "Predicted Churn": "Yes" if churn_prediction == 1 else "No",
        "Risk Category": risk,
        "Predicted Monthly Charges": round(monthly_charge_prediction, 2)
    }

# --- Streamlit UI ---
st.title("Customer Prediction App")

st.write("Enter customer data to get churn and monthly charge predictions.")

# Example of how you might collect input (this would be much more detailed for a real app)
# You'll need to create input fields for all features your models expect.

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)
# ... other input fields for all features ...

# Create a dummy customer_data_dict for demonstration
# YOU WILL NEED TO REPLACE THIS WITH ACTUAL INPUTS FROM YOUR UI
# Make sure all features expected by your models are present and correctly formatted.
customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': tenure,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': monthly_charges,
    'TotalCharges': 100.0 # This will be dropped for linear model, but needed for logistic preprocessing
}

if st.button("Predict"):    
    prediction_result = predict_customer_for_streamlit(customer_data)
    
    if "error" in prediction_result:
        st.error(prediction_result["error"])
    else:
        st.subheader("Prediction Results:")
        st.write(f"Churn Probability: {prediction_result['Churn Probability (%)']:.2f}%")
        st.write(f"Predicted Churn: {prediction_result['Predicted Churn']}")
        st.write(f"Risk Category: {prediction_result['Risk Category']}")
        st.write(f"Predicted Monthly Charges: ${prediction_result['Predicted Monthly Charges']:.2f}")
