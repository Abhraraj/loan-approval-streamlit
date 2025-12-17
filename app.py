import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()

# Always take expected columns from model
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

# Credit Info
st.subheader("Credit Details")
credit_history = st.selectbox("Credit History (1 = Good, 0 = Poor)", [1, 0])
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Check Loan Status"):

    # Rule-based check (simple, safe)
    if cibil_score < 500:
        st.error("Loan Rejected: CIBIL score below 500.")
    else:
        # Build input dictionary
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

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Align columns EXACTLY with training schema
        input_df = input_df.reindex(columns=EXPECTED_COLUMNS)

        # ML Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("Loan Approved")
            st.write(f"Approval Probability: {probability:.2%}")
        else:
            st.error("Loan Rejected")
            st.write(f"Approval Probability: {probability:.2%}")
