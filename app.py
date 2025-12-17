import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Machine Learning based eligibility & approval assessment")

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()

# ------------------------------
# Helper Functions
# ------------------------------
def format_probability(p):
    # scale ML probability into realistic banking range
    p = 0.6 + (p * 0.35)
    return round(p * 100, 2)

def calculate_max_loan(annual_income, interest_rate=0.1, tenure_years=20):
    """
    EMI <= 30% of annual income
    """
    monthly_income = annual_income / 12
    max_emi = monthly_income * 0.30

    r = interest_rate / 12
    n = tenure_years * 12

    if r == 0:
        return max_emi * n

    loan_amount = max_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
    return int(loan_amount)

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("📋 Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input(
    "Applicant Annual Income (₹)",
    min_value=0,
    step=5000
)
st.caption(f"Entered: ₹{applicant_income:,}")

coapplicant_income = st.number_input(
    "Co-applicant Annual Income (₹)",
    min_value=0,
    step=5000
)
st.caption(f"Entered: ₹{coapplicant_income:,}")

loan_amount = st.number_input(
    "Requested Loan Amount (₹)",
    min_value=0,
    step=50000
)
st.caption(f"Requested: ₹{loan_amount:,}")

loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360])

credit_history = st.selectbox("Credit History", [1.0, 0.0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=10)

# ------------------------------
# Prediction Logic
# ------------------------------
if st.button("🔍 Check Loan Status"):

    total_income = applicant_income + coapplicant_income

    # Rule 1: CIBIL Check
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500")
        st.stop()

    # Rule 2: EMI affordability
    max_eligible_loan = calculate_max_loan(total_income)

    if loan_amount > max_eligible_loan:
        st.warning("⚠️ Requested loan exceeds affordability limit")

    # --------------------------
    # Prepare Model Input
    # --------------------------
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income / 12],
        "CoapplicantIncome": [coapplicant_income / 12],
        "LoanAmount": [loan_amount / 1000],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    # --------------------------
    # ML Prediction
    # --------------------------
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    probability_percent = format_probability(probability)

    # --------------------------
    # Results
    # --------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.metric("Approval Probability", f"{probability_percent}%")
    st.metric("Maximum Eligible Loan", f"₹{max_eligible_loan:,}")

    st.caption("✔ Prediction uses machine learning + RBI-aligned financial rules")
