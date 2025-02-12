import pickle

import pandas as pd
import streamlit as st

model = pickle.load(open("random_forest_model.pkl", "rb"))


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


# Custom CSS for styling
st.markdown("""
    <style>
    /* Full page background color */
    .stApp {
        background-color: #066BB0;
    }

    /* Main title styling */
    .main-title-home {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 16px;
        text-align: center;
        color: white; /* Change text color to white for better visibility */
    }
    .main-title-features {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 16px;
        margin-top: 90px;
        text-align: center;
        color: white; /* Change text color to white for better visibility */
    }
    .main-title-form {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 16px;
        margin-top: 40px;
        text-align: center;
        color: white; /* Change text color to white for better visibility */
    }

    /* Subtitle styling */
    .subtitle-home {
        font-size: 16px;
        color: white; /* Change text color to white for better visibility */
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle-features {
        font-size: 16px;
        color: white; /* Change text color to white for better visibility */
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle-form {
        font-size: 16px;
        color: white; /* Change text color to white for better visibility */
        text-align: center;
        margin-bottom: 20px;
    }

    /* Feature card styling */
    .feature-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        text-align: center;
        background-color: white; /* Add background color to feature cards */
    }

    /* Feature title styling */
    .feature-title-analytics {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #066BB0;
    }
    .feature-title-security {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #066BB0;
    }
    .feature-title-monitoring {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #066BB0;
    }

    /* Feature description styling */
    .feature-description {
        font-size: 14px;
        color: #666;
    }

    /* Form styling */
    .form-box {

        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white; /* Add background color to form box */
    }
    .main-title, .subtitle, label {
        color: white !important;
        font-weight: bold;
    }

    /* Add a white border to input fields */
    .stTextInput input, .stNumberInput input {
        border: 2px solid white !important;
        background-color: transparent !important;
    }

    /* Placeholder text color */
    ::placeholder {
        opacity: 1;
    }
    /* Input field styling */
    .stTextInput input, .stNumberInput input {
        width: 100%;
        padding: 10px;
    }

    /* Button styling */
    .stButton button {
        width: 20%;
        padding: 10px;
        text-align: center;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: auto; /* Center horizontally */
    }

    .stButton>button {
        width: 20%;
        padding: 10px;
        text-align: center;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: auto; /* Center horizontally */
    }

    </style>
    """, unsafe_allow_html=True)

# Page 1: Introduction (Image 1)
st.markdown('<div class="main-title-home">Credit Risk Assessment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-home">Advanced machine learning for precise credit risk assessment. Make informed lending decisions with confidence.</div>',
    unsafe_allow_html=True)
if st.button("Start Assessment", key="start_assessment"):
    st.session_state['page'] = 'features'

# Page 2: Features (Image 2)
if 'page' in st.session_state and st.session_state['page'] == 'features':
    st.markdown('<div class="main-title-features">Powerful Features for Risk Assessment</div>', unsafe_allow_html=True)
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title-analytics">Advanced Analytics</div>
                <div class="feature-description">Leverage machine learning for accurate risk predictions.</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title-security">Secure Platform</div>
                <div class="feature-description">Bank-grade security for your sensitive data.</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title-monitoring">Credit Monitoring</div>
                <div class="feature-description">Real-time monitoring of credit portfolios.</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="main-title-features">Ready to Transform Your Risk Assessment?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-features">Join leading financial institutions using our advanced credit risk assessment platform.</div>',
        unsafe_allow_html=True)
    if st.button("Get Started", key="get_started"):
        st.session_state['page'] = 'form'

if 'page' in st.session_state and st.session_state['page'] == 'form':
    st.markdown('<div class="main-title-form">Enter Applicant Details</div>', unsafe_allow_html=True)

    with st.form("credit_risk_form"):
        # Removed the Full Name input field
        age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        owns_car = st.selectbox("Owns Car", ["Yes", "No"], key="owns_car")
        owns_house = st.selectbox("Owns House", ["Yes", "No"], key="owns_house")
        no_of_children = st.number_input("Number of Dependants", min_value=0, value=0, key="no_of_children")
        net_yearly_income = st.number_input("Net Year Income ($)", min_value=0.0, value=50000.0, key="net_yearly_income")
        no_of_days_employed = st.number_input("Number of Days Employed", min_value=0, value=365,
                                              key="no_of_days_employed")
        occupation_type = st.selectbox("Occupation Type", label_encoder["occupation_type"].classes_,
                                       key="occupation_type")
        total_family_members = st.number_input("Total Family Members", min_value=1, value=2, key="total_family_members")
        migrant_worker = st.selectbox("Migrant Worker", ["Yes", "No"], key="migrant_worker")
        yearly_debt_payments = st.number_input("Yearly Debt Payments", min_value=0.0, value=10000.0,
                                               key="yearly_debt_payments")
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=10000.0, key="credit_limit")
        credit_limit_used = st.number_input("Credit Limit Used (%)", min_value=0, max_value=100, value=30,
                                            key="credit_limit_used")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, key="credit_score")
        prev_defaults = st.number_input("Previous Defaults", min_value=0, value=0, key="prev_defaults")
        default_in_last_6months = st.number_input("Default in Last 6 Months", min_value=0, value=0,
                                                  key="default_in_last_6months")

        submitted = st.form_submit_button("Predict Credit Risk")

        if submitted:
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

            # Make the prediction
            binary_prediction = model.predict(input_df)
            prediction_label = "Yes" if binary_prediction[0] == 1 else "No"

            st.success("Form submitted successfully!")
            st.subheader("Prediction Result")
            # Change color based on prediction
            if prediction_label == "Yes":
                st.markdown(
                    f'<p style="color: green; font-size: 20px; font-weight: bold;">Prediction for credit card default: <strong>{prediction_label}</strong></p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p style="color: red; font-size: 20px; font-weight: bold;">Prediction for credit card default: <strong>{prediction_label}</strong></p>',
                    unsafe_allow_html=True
                )