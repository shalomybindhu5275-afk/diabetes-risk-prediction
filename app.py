import streamlit as st
import pandas as pd
import joblib
import os


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="DiabetesAI",
    page_icon="🩺",
    layout="wide"
)


# -------------------------------
# Load Model and Scaler
# -------------------------------

model = joblib.load("diabetes_random_forest.pkl")
scaler = joblib.load("scaler.pkl")


# -------------------------------
# Title
# -------------------------------

st.title("🩺 DiabetesAI")
st.subheader("Smart Diabetes Risk Prediction & Health Data Analytics")

st.write(
    "Enter your health-related information below to get "
    "a machine learning based diabetes risk estimate."
)

st.divider()


# -------------------------------
# User Input
# -------------------------------

st.header("📋 Enter Your Health Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


# -------------------------------
# Prediction
# -------------------------------

st.divider()

if st.button("🔮 Predict Diabetes Risk", use_container_width=True):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    risk_percentage = probability * 100


    # -------------------------------
    # Display Result
    # -------------------------------

    st.subheader("📊 Prediction Result")

    st.metric(
        "Estimated Diabetes Risk",
        f"{risk_percentage:.2f}%"
    )

    if prediction[0] == 1:
        st.error("⚠️ Model Prediction: Higher Diabetes Risk")
    else:
        st.success("✅ Model Prediction: Lower Diabetes Risk")


    # -------------------------------
    # Input Summary
    # -------------------------------

    st.subheader("🔎 Your Input Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    summary_col1.metric("Glucose", glucose)
    summary_col2.metric("BMI", bmi)
    summary_col3.metric("Blood Pressure", blood_pressure)
    summary_col4.metric("Age", age)

    # -------------------------------
# Dataset Analytics
# -------------------------------

st.divider()

st.header("📈 Dataset Analytics")

data = pd.read_csv("diabetes.csv")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", data.shape[0])
col2.metric("Diabetes Cases", int(data["Outcome"].sum()))
col3.metric("No Diabetes Cases", int((data["Outcome"] == 0).sum()))

st.subheader("🩺 Diabetes Outcome Distribution")

outcome_data = data["Outcome"].value_counts().rename(
    index={0: "No Diabetes", 1: "Diabetes"}
)

st.bar_chart(outcome_data)

    # -------------------------------
# Model Comparison
# -------------------------------

st.divider()

st.markdown(
    "<h2 style='text-align: center;'>🤖 Machine Learning Model Comparison</h2>",
    unsafe_allow_html=True
)

comparison_data = {
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        70.78,
        75.32,
        68.18,
        77.92,
        74.03
    ]
}

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)
st.subheader("📊 Accuracy Comparison")

st.markdown(
    "<h3 style='text-align: center;'>📊 Accuracy Comparison</h3>",
    unsafe_allow_html=True
)

st.bar_chart(
    comparison_df.set_index("Model")["Accuracy"]
)

st.write(
    "Random Forest achieved the highest accuracy among the "
    "five tested machine learning models."
)


# -------------------------------
# Disclaimer
# -------------------------------

st.divider()

st.info(
    "⚠️ This application is created for educational and project "
    "demonstration purposes. The result is a machine learning "
    "prediction and should not be considered a medical diagnosis."
)