import streamlit as st
import joblib
import pandas as pd

model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter health details to get an educational ML prediction.")
st.warning("This project is for education only. It is not medical advice or a diagnosis.")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=1, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=1, max_value=200, value=72)
skin_thickness = st.number_input("Skin Thickness", min_value=1, max_value=100, value=25)
insulin = st.number_input("Insulin", min_value=1, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0)
pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Risk"):
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age,
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Higher predicted diabetes risk: {probability:.1%}")
    else:
        st.success(f"Lower predicted diabetes risk: {probability:.1%}")

    st.caption("Please consult a qualified healthcare professional for medical concerns.")