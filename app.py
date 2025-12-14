import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_model.pkl")

model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval status.")

st.markdown("---")

# -------------------------------
# Input fields
# -------------------------------
no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    step=1
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    step=10000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    step=10000
)

loan_term = st.number_input(
    "Loan Term",
    min_value=0,
    step=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    step=1
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    step=10000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    step=10000
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    step=10000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    step=10000
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Loan Approval"):
    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success(f"✅ Loan Approved (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Loan Rejected (Approval Probability: {probability:.2%})")

    st.subheader("Entered Details")
    st.dataframe(input_df)
