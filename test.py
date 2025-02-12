import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load the model from the .h5 file
model = tf.keras.models.load_model('ann_model.h5')

# Load the label encoder and scaler from pickle files
with open('encoder.pickle', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open('scaler.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the input features with all necessary columns
def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=100, value=35)
    gender = st.selectbox('Gender', ['M', 'F'])
    owns_car = st.selectbox('Owns Car', ['Y', 'N'])
    owns_house = st.selectbox('Owns House', ['Y', 'N'])
    no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=2)
    net_yearly_income = st.number_input('Net Yearly Income', min_value=0, value=109862)
    no_of_days_employed = st.number_input('Number of Days Employed', min_value=0, value=1000)
    occupation_type = st.selectbox('Occupation Type', ['Unknown', 'Laborer', 'Core staff', 'Sales staff', 'Accountants', 'Managers', 'Drivers', 'HR staff', 'IT staff', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'High skill tech staff', 'Waiters/barmen staff', 'Realty agents', 'Low-skill Laborers', 'Sales Agents', 'Secretaries', 'Other'])
    total_family_members = st.number_input('Total Family Members', min_value=1, max_value=20, value=4)
    migrant_worker = st.selectbox('Migrant Worker', [0.0, 1.0])
    yearly_debt_payments = st.number_input('Yearly Debt Payments', min_value=0, value=10000)
    credit_limit = st.number_input('Credit Limit', min_value=0, value=10000)
    credit_limit_used = st.number_input('Credit Limit Used (%)', min_value=0, max_value=100, value=30)
    credit_score = st.number_input('Credit Score', min_value=0, value=700)
    prev_defaults = st.number_input('Previous Defaults', min_value=0, value=0)
    default_in_last_6months = st.selectbox('Default in Last 6 Months', [0, 1])

    data = {
        'age': age,
        'gender': gender,
        'owns_car': owns_car,
        'owns_house': owns_house,
        'no_of_children': no_of_children,
        'net_yearly_income': net_yearly_income,
        'no_of_days_employed': no_of_days_employed,
        'occupation_type': occupation_type,
        'total_family_members': total_family_members,
        'migrant_worker': migrant_worker,
        'yearly_debt_payments': yearly_debt_payments,
        'credit_limit': credit_limit,
        'credit_limit_used(%)': credit_limit_used,
        'credit_score': credit_score,
        'prev_defaults': prev_defaults,
        'default_in_last_6months': default_in_last_6months
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Define the expected data types
data_types = {
    'age': 'int64',
    'gender': 'int32',
    'owns_car': 'int32',
    'owns_house': 'int32',
    'no_of_children': 'float64',
    'net_yearly_income': 'float64',
    'no_of_days_employed': 'float64',
    'occupation_type': 'int32',
    'total_family_members': 'float64',
    'migrant_worker': 'float64',
    'yearly_debt_payments': 'float64',
    'credit_limit': 'float64',
    'credit_limit_used(%)': 'int64',
    'credit_score': 'float64',
    'prev_defaults': 'int64',
    'default_in_last_6months': 'int64'
}

# Encode categorical features using the label encoder
categorical_features = ['gender', 'owns_car', 'owns_house', 'occupation_type']
for feature in categorical_features:
    input_df[feature] = label_encoder[feature].transform(input_df[feature])

# Set the data types for the DataFrame
input_df = input_df.astype(data_types)

# Define numerical features for scaling
numerical_features = scaler.feature_names_in_

# Scale the numerical features using the scaler
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

# Make the prediction
prediction = model.predict(input_df)

# Convert the prediction to binary classification (0 or 1)
binary_prediction = (prediction > 0.5).astype(int)

# Label the prediction as "Yes" or "No"
prediction_label = "Yes" if binary_prediction[0][0] == 1 else "No"

# Output the prediction
st.subheader('Prediction')
st.write(f'Prediction for credit card default: {prediction_label}')