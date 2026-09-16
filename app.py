import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Page Configuration
st.set_page_config(
    page_title="Heart Guard AI",
    layout="wide"
)

# 2. Title & Team/Tools Subheader
st.title("HeartGuard — AI Clinical Risk Assessor")

# إضافة أسماء التيم والأدوات بخط صغير تحت العنوان مباشرة
st.markdown(
    """
    <p style='color: #888888; font-size: 14px; margin-top: -10px; margin-bottom: 20px;'>
    <b>Team:</b> Mohamed Salah, Abdelmoniem Elsayed, Mariam Mahmoud, Rahma Maged, Dai Alaa &nbsp;|&nbsp; 
    <b>Tools:</b> Python, Streamlit, Scikit-Learn, Pandas, NumPy, Logistic Regression
    </p>
    """,
    unsafe_allow_html=True
)

user_role = st.radio(
    "Select User Profile:",
    options=["Standard User (Patient Profile)", "Medical Specialist (Doctor Profile)"],
    horizontal=True
)

st.markdown("---")

# 3. Data Loading & Model Training
@st.cache_resource
def train_model():
    # Data Loading
    df = pd.read_csv("HeartDiseaseTrain-Test.csv")

    # Duplicates Removal
    df = df.drop_duplicates()

    # Categorical Encoding
    df_encoded = pd.get_dummies(
        df,
        columns=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'],
        drop_first=True,
        dtype=int
    )

    # Train Test Split
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model Training
    logistic_model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)

    return logistic_model, scaler, X.columns.tolist()

logistic_model, scaler, feature_columns = train_model()

# 4. Inputs Form
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
    raw_input = {
        'age': age,
        'sex': sex,
        'chest_pain_type': cp,
        'resting_blood_pressure': trestbps,
        'cholestoral': chol,
        'fasting_blood_sugar': 1 if fbs == "True" else 0,
        'rest_ecg': restecg,
        'max_heart_rate_achieved': thalach,
        'exercise_induced_angina': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'vessels_colored_by_flourosopy': ca,
        'thalassemia': thal
    }
    
    input_df_raw = pd.DataFrame([raw_input])
    
    input_encoded = pd.get_dummies(
        input_df_raw,
        columns=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'],
        drop_first=True,
        dtype=int
    )
    
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    input_scaled = scaler.transform(input_final)
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
