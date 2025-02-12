import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load the model, label encoder, and scaler
model = tf.keras.models.load_model("ann_model.h5")

with open("encoder.pickle", "rb") as le_file:
    label_encoder = pickle.load(le_file)

with open("scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the expected data types
data_types = {
    "age": "int64",
    "gender": "int32",
    "owns_car": "int32",
    "owns_house": "int32",
    "no_of_children": "float64",
    "net_yearly_income": "float64",
    "no_of_days_employed": "float64",
    "occupation_type": "int32",
    "total_family_members": "float64",
    "migrant_worker": "float64",
    "yearly_debt_payments": "float64",
    "credit_limit": "float64",
    "credit_limit_used(%)": "int64",
    "credit_score": "float64",
    "prev_defaults": "int64",
    "default_in_last_6months": "int64",
}

# Define categorical and numerical features
categorical_features = ["gender", "owns_car", "owns_house", "occupation_type"]
numerical_features = scaler.feature_names_in_

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.selectbox("Select a page", ["Home", "Credit Risk Assessment"])

# Home page
def display_home():
    st.title("Credit Risk Assessment Web App")
    st.markdown(
        """
        Welcome to the Credit Risk Assessment Web App! This tool evaluates creditworthiness based on multiple financial factors, helping financial institutions make data-driven lending decisions.

        **Key Features:**
        - Evaluate risk associated with loan applicants
        - Analyze financial behavior and debt usage
        - Identify potential defaulters
        - Improve decision-making for financial institutions

        Enter the applicant's details below to assess credit risk.
        """
    )

# Credit Risk Assessment page
def display_credit_risk_assessment():
    st.title("Credit Risk Assessment")

    # Input fields for user data
    st.subheader("Enter Applicant Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    owns_car = st.selectbox("Owns Car", ["Yes", "No"])
    owns_house = st.selectbox("Owns House", ["Yes", "No"])
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    net_yearly_income = st.number_input("Net Yearly Income", min_value=0.0, value=50000.0)
    no_of_days_employed = st.number_input("Number of Days Employed", min_value=0, value=365)
    occupation_type = st.selectbox("Occupation Type", label_encoder["occupation_type"].classes_)
    total_family_members = st.number_input("Total Family Members", min_value=1, value=2)
    migrant_worker = st.selectbox("Migrant Worker", ["Yes", "No"])
    yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0.0, value=10000.0)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=10000.0)
    credit_limit_used = st.number_input("Credit Limit Used (%)", min_value=0, max_value=100, value=30)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    prev_defaults = st.number_input("Previous Defaults", min_value=0, value=0)
    default_in_last_6months = st.number_input("Default in Last 6 Months", min_value=0, value=0)

    # Convert categorical selections to expected format
    gender_encoded = "F" if gender == "Female" else "M"
    owns_car_encoded = "Y" if owns_car == "Yes" else "N"
    owns_house_encoded = "Y" if owns_house == "Yes" else "N"
    migrant_worker_encoded = 1.0 if migrant_worker == "Yes" else 0.0

    # Create a dictionary of input features
    input_features = {
        "age": age,
        "gender": gender_encoded,
        "owns_car": owns_car_encoded,
        "owns_house": owns_house_encoded,
        "no_of_children": float(no_of_children),
        "net_yearly_income": float(net_yearly_income),
        "no_of_days_employed": float(no_of_days_employed),
        "occupation_type": occupation_type,
        "total_family_members": float(total_family_members),
        "migrant_worker": migrant_worker_encoded,
        "yearly_debt_payments": float(yearly_debt_payments),
        "credit_limit": float(credit_limit),
        "credit_limit_used(%)": credit_limit_used,
        "credit_score": float(credit_score),
        "prev_defaults": prev_defaults,
        "default_in_last_6months": default_in_last_6months,
    }

    # Convert the input features to a DataFrame
    input_df = pd.DataFrame([input_features])

    # Encode categorical features
    for feature in categorical_features:
        input_df[feature] = label_encoder[feature].transform(input_df[feature])

    # Set the data types for the DataFrame
    input_df = input_df.astype(data_types)

    # Scale the numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Make the prediction when the button is clicked
    if st.button("Predict Credit Risk"):
        prediction = model.predict(input_df)
        binary_prediction = (prediction > 0.5).astype(int)
        prediction_label = "Yes" if binary_prediction[0][0] == 1 else "No"
        st.subheader("Prediction Result")
        st.write(f"Prediction for credit card default: **{prediction_label}**")

# Display the selected page
if options == "Home":
    display_home()
elif options == "Credit Risk Assessment":
    display_credit_risk_assessment()
