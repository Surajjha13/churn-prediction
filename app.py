import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("catboost_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = float(open("threshold.txt").read())

st.title("Customer Churn Prediction App")

# User input fields
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 90, 40)
tenure = st.number_input("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0,1])
is_active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
gender = st.selectbox("Gender", ["Male","Female"])
geo = st.selectbox("Geography", ["France","Germany","Spain"])

# Convert to model input
# (same feature engineering used during training)
input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': has_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary,
    'BalanceSalaryRatio': balance / (salary + 1),
    'AgeTenureRatio': age / (tenure + 1),
    'ProductsPerTenure': products / (tenure + 1),
    'HasBalanceFlag': 1 if balance > 0 else 0,
    'IsSeniorCitizen': 1 if age > 60 else 0,
    'CreditScoreBin': 0 if credit_score < 500 else 1 if credit_score < 650 else 2,
    'Geography_Germany': 1 if geo=="Germany" else 0,
    'Geography_Spain': 1 if geo=="Spain" else 0,
    'Gender_Male': 1 if gender=="Male" else 0
}

input_df = pd.DataFrame([input_dict])

# Scale
scaled_input = scaler.transform(input_df)

# Predict
prob = model.predict_proba(scaled_input)[:, 1][0]
pred = 1 if prob >= threshold else 0

# Display results in a more helpful way
st.write("## Prediction Result")

if pred == 1:
    st.error("### High Churn Risk Detected")
    st.write(f"**Probability of Churn:** `{prob:.2%}`")
    st.write("""
    **Interpretation:**  
    This customer has a **high likelihood of leaving the service**.  
    Immediate retention actions are recommended:
    - Offer discounts or loyalty rewards  
    - Reach out via customer support  
    - Analyze recent interactions  
    """)
else:
    st.success("### Customer is Not Likely to Churn")
    st.write(f"**Probability of Churn:** `{prob:.2%}`")
    st.write("""
    **Interpretation:**  
    The customer is **stable and unlikely to leave** at the moment.  
    Regular engagement and satisfaction monitoring is sufficient.
    """)

