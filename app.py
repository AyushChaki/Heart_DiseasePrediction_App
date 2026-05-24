import streamlit as st
# This MUST be the first streamlit line!
st.set_page_config(page_title="My App", layout="wide")
import pandas as pd
import joblib
model=joblib.load("KNN_heart_model.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")


st.title("Heart Disease Prediction App")
st.markdown("""
### Project Description
This machine learning web application predicts the likelihood of heart disease based on medical attributes provided by the user.

### Model Information
- Model Used: K-Nearest Neighbors (KNN)
- Machine Learning Library: Scikit-learn
- Data Preprocessing: StandardScaler
- Accuracy: 86% 

### GitHub Repository
[View Source Code](https://github.com/AyushChaki/AYUSH_ML)
""")
st.markdown("Enter the following details to predict:")
age=st.slider("Age", 18, 100, 40)
sex=st.selectbox("Sex", options=["Male", "Female"])
cp=st.selectbox("Chest Pain Type", options=["TA", "ATA", "NAP", "ASY"])
p=st.number_input("Resting Blood Pressure(mm Hg)", 80, 200, 120)
chol=st.number_input("Cholesterol(mg/dl)", 100, 600, 200)
fbs=st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
resting_ecg=st.selectbox("Resting ECG Results", options=["Normal", "ST", "LVH"])
max_hr=st.slider("Maximum Heart Rate", 60, 220, 150)
ex_ang=st.selectbox("Exercise Induced Angina",["No", "Yes"])
oldpeak=st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope=st.selectbox("ST Slope", options=["Up", "Flat", "Downs"])
if st.button("Predict"):
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    ex_ang = 1 if ex_ang == "Yes" else 0
    raw_input = {
        "Age": age,
        "RestingBP": p,
        "Cholesterol": chol,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": sex,
        "FastingBS": fbs,
        "ExerciseAngina_Y": ex_ang,

        "ChestPainType_ATA": 1 if cp == "ATA" else 0,
        "ChestPainType_NAP": 1 if cp == "NAP" else 0,
        "ChestPainType_TA": 1 if cp == "TA" else 0,

        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

        "ST_Slope_Flat": 1 if slope == "Flat" else 0,
        "ST_Slope_Up": 1 if slope == "Up" else 0
    }
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns]
    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)[0]  
    if prediction==1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")     
