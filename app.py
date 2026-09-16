import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Page Configuration
st.set_page_config(
    page_title="HeartGuard AI — Clinical Risk Assessor",
    layout="wide"
)

# 2. Pipeline Training & Caching
@st.cache_resource
def load_and_train_pipeline():
    df = pd.read_csv("HeartDiseaseTrain-Test.csv")
    df = df.drop_duplicates()

    df_encoded = pd.get_dummies(
        df,
        columns=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 
                 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'],
        drop_first=True,
        dtype=int
    )

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()

logistic_model, scaler, feature_columns = load_and_train_pipeline()

# 3. Header & User Profile Selection
st.title("HeartGuard — AI Clinical Risk Assessor")
st.write("Select your access profile to adjust the required clinical input level.")

user_role = st.radio(
    "Select User Profile:",
    options=["Standard User (Patient Profile)", "Medical Specialist (Doctor Profile)"],
    horizontal=True
)

st.markdown("---")

# 4. Input Form Construction
with st.form("risk_assessment_form"):
    st.subheader("Patient Baseline Attributes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=230, value=120)
    
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["False", "True"])

    if "Specialist" in user_role:
        st.markdown("---")
        st.subheader("Advanced Diagnostic Features (Specialist Mode)")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
            restecg = st.selectbox("Resting ECG Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
            
        with adv_col2:
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Segment Slope", options=["Upsloping", "Flat", "Downsloping"])
            ca = st.slider("Major Vessels Colored by Fluoroscopy", 0, 4, 0)
            thal = st.selectbox("Thalassemia Status", options=["Normal", "Fixed Defect", "Reversible Defect"])
    else:
        cp = "Typical Angina"
        restecg = "Normal"
        thalach = 150
        exang = "No"
        oldpeak = 0.0
        slope = "Upsloping"
        ca = 0
        thal = "Normal"

    submit_button = st.form_submit_button("Run Risk Prediction")

# 5. Prediction Logic
if submit_button:
    input_dict = {col: 0 for col in feature_columns}

    for col_name in input_dict.keys():
        if "age" in col_name.lower(): input_dict[col_name] = age
        if "resting" in col_name.lower() or "trestbps" in col_name.lower(): input_dict[col_name] = trestbps
        if "chol" in col_name.lower(): input_dict[col_name] = chol
        if "max" in col_name.lower() or "thalach" in col_name.lower(): input_dict[col_name] = thalach
        if "oldpeak" in col_name.lower(): input_dict[col_name] = oldpeak

    if sex == "Male":
        for col in input_dict:
            if "sex" in col.lower() and ("1" in col or "male" in col): input_dict[col] = 1

    if fbs == "True":
        for col in input_dict:
            if "fasting" in col.lower() and "1" in col: input_dict[col] = 1

    if exang == "Yes":
        for col in input_dict:
            if "exercise" in col.lower() and "1" in col: input_dict[col] = 1

    if cp == "Atypical Angina":
        for col in input_dict:
            if "chest" in col.lower() and "1" in col: input_dict[col] = 1
    elif cp == "Non-Anginal Pain":
        for col in input_dict:
            if "chest" in col.lower() and "2" in col: input_dict[col] = 1
    elif cp == "Asymptomatic":
        for col in input_dict:
            if "chest" in col.lower() and "3" in col: input_dict[col] = 1

    if restecg == "ST-T Wave Abnormality":
        for col in input_dict:
            if "rest" in col.lower() and "1" in col: input_dict[col] = 1
    elif restecg == "Left Ventricular Hypertrophy":
        for col in input_dict:
            if "rest" in col.lower() and "2" in col: input_dict[col] = 1

    if slope == "Flat":
        for col in input_dict:
            if "slope" in col.lower() and "1" in col: input_dict[col] = 1
    elif slope == "Downsloping":
        for col in input_dict:
            if "slope" in col.lower() and "2" in col: input_dict[col] = 1

    if ca > 0:
        for col in input_dict:
            if "vessels" in col.lower() and str(ca) in col: input_dict[col] = 1

    if thal == "Fixed Defect":
        for col in input_dict:
            if "thalassemia" in col.lower() and "1" in col: input_dict[col] = 1
    elif thal == "Reversible Defect":
        for col in input_dict:
            if "thalassemia" in col.lower() and "2" in col: input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])[feature_columns]

    input_scaled = scaler.transform(input_df)
    prediction = logistic_model.predict(input_scaled)[0]
    probability = logistic_model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    st.subheader("Diagnostic Outcome")
    
    if prediction == 1:
        st.error(f"High Cardiovascular Risk Detected! Risk Probability: {probability:.1f}%")
        if "Standard" in user_role:
            st.info("Recommendation: Please consult a cardiologist for further diagnostic evaluation.")
        else:
            st.warning("Clinical Note: Positive classification based on trained feature thresholds.")
    else:
        st.success(f"Low Cardiovascular Risk. Risk Probability: {probability:.1f}%")
        st.info("Recommendation: Maintain a healthy lifestyle and schedule annual medical checkups.")
