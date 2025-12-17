import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()
EXPECTED_COLUMNS = list(model.feature_names_in_)

# ---------------- UI ----------------
st.title("Loan Approval Prediction System")
st.write("Machine Learning based Loan Eligibility Checker")

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

# Credit Details
st.subheader("Credit Profile")
credit_history = st.selectbox("Credit History (1 = Good, 0 = Poor)", [1, 0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

st.markdown("---")

# ---------------- BUTTON + LOGIC ----------------
if st.button("Check Loan Status"):

    # Rule 1: CIBIL
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500.")
    
    else:
        total_income = applicant_income + coapplicant_income

        if total_income == 0:
            st.error("❌ Loan Rejected: Total income cannot be zero.")
        else:
            # Rule 2: EMI ≤ 30% income
            max_annual_emi = total_income * 0.30
            max_monthly_emi = max_annual_emi / 12

            # Standard assumptions
            interest_rate = 10
            tenure_months = loan_term * 12
            monthly_rate = interest_rate / (12 * 100)

            requested_emi = (
                loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months
            ) / ((1 + monthly_rate) ** tenure_months - 1)

            # Rule 3: Max eligible loan
            eligible_loan_amount = (
                max_monthly_emi * ((1 + monthly_rate) ** tenure_months - 1)
            ) / (monthly_rate * (1 + monthly_rate) ** tenure_months)

            if requested_emi > max_monthly_emi:
                st.warning("⚠️ Loan Amount Not Affordable")
                st.write(f"Maximum Eligible Loan Amount: ₹{int(eligible_loan_amount):,}")
            else:
                # ML input
                input_data = {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": tenure_months,
                    "Credit_History": credit_history,
                    "Property_Area": property_area
                }

                input_df = pd.DataFrame([input_data])
                input_df = input_df.reindex(columns=EXPECTED_COLUMNS)

                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                if prediction == 1:
                    st.success("✅ Loan Approved")
                    st.write(f"Approval Probability (ML): {probability:.2%}")
                    st.write(f"Eligible Loan Amount: ₹{int(eligible_loan_amount):,}")
                else:
                    st.error("❌ Loan Rejected by Risk Model")
                    st.write(f"Approval Probability (ML): {probability:.2%}")
