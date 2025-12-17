import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def calculate_emi(principal, annual_rate, tenure_years):
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    emi = principal * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
    return emi

def max_loan_from_emi(max_emi, annual_rate, tenure_years):
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    loan = max_emi * (((1 + monthly_rate) ** months) - 1) / (monthly_rate * ((1 + monthly_rate) ** months))
    return loan

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎓 Loan Approval Prediction System")
st.subheader("Machine Learning + Rule-Based Risk Controls")

st.markdown("---")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
income = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0)
loan_term = st.slider("Loan Term (Years)", 1, 20, 10)
credit_history = st.selectbox("Credit History (1 = Good, 0 = Poor)", [1, 0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

interest_rate = 10  # fixed 10% (industry standard)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------
if st.button("Check Loan Eligibility"):
    
    # ---------- Rule 1: CIBIL Check ----------
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500 is not eligible for loan approval.")
    
    else:
        # ---------- Rule 2: EMI Affordability ----------
        max_annual_emi = income * 0.30
        max_monthly_emi = max_annual_emi / 12
        
        eligible_loan = max_loan_from_emi(max_monthly_emi, interest_rate, loan_term)
        requested_emi = calculate_emi(loan_amount, interest_rate, loan_term)
        
        if requested_emi > max_monthly_emi:
            st.warning("⚠️ Loan Amount Too High")
            st.write(f"Maximum Eligible Loan Amount: ₹{int(eligible_loan):,}")
            st.write("Reason: EMI exceeds 30% of annual income.")
        
        else:
            # ---------- ML Prediction ----------
            input_data = pd.DataFrame({
                "ApplicantIncome": [income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_term * 12],
                "Credit_History": [credit_history]
            })
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # ---------- Final Decision ----------
            if prediction == 1:
                st.success("✅ Loan Approved")
                st.write(f"Approval Probability: {probability:.2%}")
                st.write(f"Eligible Loan Amount: ₹{int(eligible_loan):,}")
                st.write(f"Estimated Monthly EMI: ₹{int(requested_emi):,}")
            else:
                st.error("❌ Loan Rejected by Risk Model")
                st.write(f"Approval Probability: {probability:.2%}")

