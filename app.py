import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Prediction")

gender = st.selectbox("Gender", [0,1])
married = st.selectbox("Married", [0,1])
dependents = st.number_input("Dependents", 0,5)
education = st.selectbox("Education", [0,1])
self_employed = st.selectbox("Self Employed", [0,1])

income = st.number_input("Applicant Income")

co_income = st.number_input("Coapplicant Income")

loan_amount = st.number_input("Loan Amount")

loan_term = st.number_input("Loan Amount Term")

credit = st.selectbox("Credit History", [0,1])

property_area = st.selectbox("Property Area", [0,1,2])

if st.button("Predict"):

    data = np.array([[gender, married, dependents, education,
                      self_employed, income, co_income,
                      loan_amount, loan_term,
                      credit, property_area]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
        