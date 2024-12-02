import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import os
import tensorflow as tf

# Streamlit page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

model_predictor = tf.keras.models.load_model('./Model/loan_approval_model.h5')
scaler = jb.load('./Model/scaler.pkl')

st.title("Hello")


st.title("Loan Approval Prediction")

# Collect user inputs
person_income = st.number_input("Enter Your Income", min_value=500, max_value=50000, value=10000)
person_age = st.number_input("Enter Your Age", min_value=500, max_value=50000, value=10000)
person_home_ownership = st.selectbox("Your Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=10000)
annual_inc = st.number_input("Annual Income", min_value=1000, max_value=100000, value=30000)
cb_person_default_on_file = st.selectbox("Default on File", ["No", "Yes"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

