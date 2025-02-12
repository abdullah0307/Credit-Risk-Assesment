import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score






# Categorical Features
categorical_features = ["gender", "owns_car", "owns_house", "occupation_type"]

# Encoding and Scaling
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

encoded_features = encoder.fit_transform(features[categorical_features])
numerical_features = features.drop(columns=categorical_features).values

X = np.hstack((numerical_features, encoded_features))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, random_state=42
)

# Train Classification Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Streamlit UI
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


def classification_model():
    st.title("Predict Credit Card Default")

    # User Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    owns_car = st.selectbox("Owns Car", ["Yes", "No"])
    owns_house = st.selectbox("Owns House", ["Yes", "No"])
    no_of_children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=2
    )
    net_yearly_income = st.number_input(
        "Net Yearly Income", min_value=10000, max_value=500000, value=109862
    )
    no_of_days_employed = st.number_input(
        "Days Employed", min_value=0, max_value=40000, value=1000
    )
    occupation_type = st.selectbox(
        "Occupation Type", sorted(data["occupation_type"].dropna().unique())
    )
    total_family_members = st.number_input(
        "Total Family Members", min_value=1, max_value=10, value=4
    )
    migrant_worker = st.selectbox("Migrant Worker", [0, 1])

    yearly_debt_payments = st.number_input(
        "Yearly Debt Payments", min_value=0, max_value=10000000000)
    credit_limit = st.number_input("Credit Limit", min_value=0, max_value=10000000000)
    credit_limit_used = st.number_input("Credit Limit Used (%)", 0, 100)

    credit_score = st.number_input("Credit Score", min_value=0, max_value=100)
    prev_defaults = st.selectbox("Previous Defaults", ["Yes", "No"])
    default_in_last_6months = st.selectbox("Default in Last 6 Months", ["Yes", "No"])

    input_data = pd.DataFrame(
        {
            "age": [age],
            "gender": ["F" if gender == "Female" else "Male"],
            "owns_car": ["Y" if owns_car == "Yes" else "N"],
            "owns_house": ["Y" if owns_house == "Yes" else "N"],
            "no_of_children": [no_of_children],
            "net_yearly_income": [net_yearly_income],
            "no_of_days_employed": [no_of_days_employed],
            "occupation_type": [occupation_type],
            "total_family_members": [total_family_members],
            "migrant_worker": [migrant_worker],
            "yearly_debt_payments": [yearly_debt_payments],
            "credit_limit": [credit_limit],
            "credit_limit_used(%)": [credit_limit_used],
            "credit_score": [credit_score],
            "prev_defaults": [1 if prev_defaults == "Yes" else 0],
            "default_in_last_6months": [1 if default_in_last_6months == "Yes" else 0],
        }
    )

    # Preprocess Input
    encoded_input = encoder.transform(input_data[categorical_features])
    numerical_input = input_data.drop(columns=categorical_features).values
    combined_input = np.hstack((numerical_input, encoded_input))
    scaled_input = scaler.transform(combined_input)

    if st.button("Predict Credit Card Default"):
        prediction = model.predict(scaled_input)
        result = "Yes" if prediction[0] == 1 else "No"
        st.write(f"### Prediction for credit card default: {result}")


# Run Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Credit Card Default Prediction"])

if page == "Home":
    display_home()
elif page == "Credit Card Default Prediction":
    classification_model()
