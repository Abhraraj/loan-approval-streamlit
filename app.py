if st.button("Check Loan Status"):

    # ---------------- Rule 1: CIBIL ----------------
    if cibil_score < 500:
        st.error("❌ Loan Rejected: CIBIL score below 500 is not eligible for loan approval.")
    
    else:
        # ---------------- Rule 2: EMI Affordability ----------------
        total_income = applicant_income + coapplicant_income

        if total_income == 0:
            st.error("❌ Loan Rejected: Total income cannot be zero.")
        else:
            # EMI rule: max 30% of annual income
            max_annual_emi = total_income * 0.30
            max_monthly_emi = max_annual_emi / 12

            # Assumptions (standard)
            interest_rate = 10  # 10% per annum
            tenure_months = loan_term * 12
            monthly_rate = interest_rate / (12 * 100)

            # EMI for requested loan
            requested_emi = (
                loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months
            ) / ((1 + monthly_rate) ** tenure_months - 1)

            # ---------------- Rule 3: Max Loan Eligibility ----------------
            eligible_loan_amount = (
                max_monthly_emi * ((1 + monthly_rate) ** tenure_months - 1)
            ) / (monthly_rate * (1 + monthly_rate) ** tenure_months)

            if requested_emi > max_monthly_emi:
                st.warning("⚠️ Loan Amount Not Affordable")
                st.write(f"Maximum Eligible Loan Amount: ₹{int(eligible_loan_amount):,}")
                st.write("Reason: EMI exceeds 30% of total annual income.")
            
            else:
                # ---------------- ML Prediction ----------------
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
                    st.write(f"Estimated Monthly EMI: ₹{int(requested_emi):,}")
                else:
                    st.error("❌ Loan Rejected by Risk Model")
                    st.write(f"Approval Probability (ML): {probability:.2%}")
