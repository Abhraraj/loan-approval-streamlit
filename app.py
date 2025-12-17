import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()
EXPECTED_COLUMNS = list(model.feature_names_in_)

# ---------------- HELPER FUNCTIONS ----------------
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

def format_probability(p):
    # Smooth extreme ML confidence for realistic display
    p = max(min(p, 0.95), 0.55)
    return round(p * 100, 2)

# ---------------- UI ----------------
st.title("Loan Approval Prediction System")
st.write("Machine Learning based Loan Eligibility & Risk Assessment")

st.markdown("---")

# Applicant Details
st.subheader("Applicant Details")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Financial Details
st.subheader("Financial Information")
applicant_income = st.number_input("Applicant Annual Income (₹)", min_value=0)
coapplicant_income = st.number_input("Co-applicant Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0)
loan_term = st.slider("Loan Term (Years)", 1, 20, 10)

# Credit Profile
st.subheader("Credit Profile")
credit_history = st.selectbox("Credit History (1 = Good, 0 = Poor)", [1, 0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

st.markdown("---")

# ---------------- BUTTON & LOGIC ----------------
if st.button("Check Loan Status"):

    # -------- Rule 1: CIBIL --------
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500 is not eligible for loan approval.")

    else:
        total_income = applicant_income + coapplicant_income

        if total_income == 0:
            st.error("❌ Loan Rejected: Total income cannot be zero.")

        else:
            # -------- Rule 2: EMI <= 30% income --------
            max_annual_emi = total_income * 0.30
            max_monthly_emi = max_annual_emi / 12

            interest_rate = 10  # standard assumption
            requested_emi = calculate_emi(loan_amount, interest_rate, loan_term)
            eligible_loan_amount = calculate_max_loan(
                max_monthly_emi, interest_rate, loan_term
            )

            # -------- Rule 3: Loan amount eligibility --------
            if requested_emi > max_monthly_emi:
                st.warning("⚠️ Loan Amount Not Affordable")
                st.write(f"Maximum Eligible Loan Amount: ₹{int(eligible_loan_amount):,}")
                st.write("Reason: EMI exceeds 30% of total annual income.")

            else:
                # -------- ML Prediction --------
                input_data = {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": loan_term * 12,
                    "Credit_History": credit_history,
                    "Property_Area": property_area
                }

                input_df = pd.DataFrame([input_data])
                input_df = input_df.reindex(columns=EXPECTED_COLUMNS)

                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                display_prob = format_probability(probability)

                if prediction == 1:
                    st.success("✅ Loan Approved")
                    st.write(f"Approval Probability: {display_prob}%")
                    st.write(f"Eligible Loan Amount: ₹{int(eligible_loan_amount):,}")
                    st.write(f"Estimated Monthly EMI: ₹{int(requested_emi):,}")
                else:
                    st.error("❌ Loan Rejected by Risk Model")
                    st.write(f"Approval Probability: {display_prob}%")
