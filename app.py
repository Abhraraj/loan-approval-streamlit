import streamlit as st
import pandas as pd
import joblib
import math

# --------------------------------------------------
# Load model (Pipeline)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def calculate_emi(principal, annual_rate, tenure_years):
    r = annual_rate / (12 * 100)
    n = tenure_years * 12
    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
    return emi

def calculate_max_loan(max_emi, annual_rate, tenure_years):
    r = annual_rate / (12 * 100)
    n = tenure_years * 12
    loan = max_emi * (((1 + r) ** n) - 1) / (r * ((1 + r) ** n))
    return loan

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🏦 Loan Approval Prediction System")
st.caption("Machine Learning–Based Eligibility & Risk Assessment")

st.markdown("---")

# ---------------- Applicant Details ----------------
st.subheader("👤 Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------- Financial Details ----------------
st.subheader("💰 Financial Details")

applicant_income = st.number_input("Applicant Annual Income (₹)", min_value=0)
coapplicant_income = st.number_input("Co-Applicant Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0)
loan_term = st.slider("Loan Tenure (Years)", 1, 20, 10)

# ---------------- Credit Profile ----------------
st.subheader("📊 Credit Profile")

credit_history = st.selectbox("Credit History", [1, 0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Check Loan Eligibility"):
    
    # Rule 1: CIBIL
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500 is not eligible.")
    
    else:
        # EMI Rule
        total_income = applicant_income + coapplicant_income
        max_annual_emi = total_income * 0.30
        max_monthly_emi = max_annual_emi / 12

        interest_rate = 10
        requested_emi = calculate_emi(loan_amount, interest_rate, loan_term)
        eligible_loan = calculate_max_loan(max_monthly_emi, interest_rate, loan_term)

        if requested_emi > max_monthly_emi:
            st.warning("⚠️ Loan Amount Not Affordable")
            st.write(f"Maximum Eligible Loan Amount: ₹{int(eligible_loan):,}")
            st.write("Reason: EMI exceeds 30% of annual income.")
        
        else:
            # ---------- ML Input (FULL feature set) ----------
            input_df = pd.DataFrame({
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [education],
                "Self_Employed": [self_employed],
                "ApplicantIncome": [applicant_income],
                "CoapplicantIncome": [coapplicant_income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_term * 12],
                "Credit_History": [credit_history],
                "Property_Area": [property_area]
            })

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.success("✅ Loan Approved")
                st.write(f"Approval Probability (ML): {probability:.2%}")
                st.write(f"Eligible Loan Amount: ₹{int(eligible_loan):,}")
                st.write(f"Estimated Monthly EMI: ₹{int(requested_emi):,}")
            else:
                st.error("❌ Loan Rejected by Risk Model")
                st.write(f"Approval Probability (ML): {probability:.2%}")
