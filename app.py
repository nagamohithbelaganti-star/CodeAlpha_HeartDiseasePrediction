import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("models/heart_disease_model.pkl")

# Page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

# Title
st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient's information below to make a prediction.")

st.divider()

# Patient inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    [0, 1],
    help="0 = Female, 1 = Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3, 4]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3, 6, 7]
)

st.divider()

# Prediction button
if st.button("🔍 Predict"):

    # Create input data
    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0][1] * 100

    # Display result
    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write(f"**Prediction Probability: {probability:.2f}%**")

    st.progress(int(probability))

    st.warning(
        "This application is an educational machine-learning project "
        "and should not be used as a medical diagnosis."
    )
   