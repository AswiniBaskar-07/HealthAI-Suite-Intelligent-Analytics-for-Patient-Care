import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# PAGE CONFIG 
st.set_page_config(
    page_title="Hospital Risk Prediction System",
    page_icon="🏥",
    layout="wide"
)

# SAFE INPUT HELPERS 
def safe_number_input(label, min_value, max_value, value, step=1):
    """
    Safe Streamlit number_input: automatically adjusts max_value if CSV value is higher.
    """
    if value < min_value:
        value = min_value
    if value > max_value:
        max_value = value
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step)

def safe_selectbox(label, options, value):
    """
    Safely select a value from options. If value not in options, use options[0].
    """
    if value not in options:
        value = options[0]
    return st.selectbox(label, options, index=options.index(value))

#LOAD MODEL
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "final_healthcare_risk_model.pkl")
    return joblib.load(model_path)

model = load_model()
expected_features = model.named_steps["prep"].feature_names_in_

# LOAD PATIENT DATA 
@st.cache_data
def load_patient_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, r"C:\Users\BaBuReDdI\Desktop\Data Science\Project\HealthCare_Data_Enhanced.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    else:
        st.warning("Patient records CSV not found. You can enter details manually.")
        return pd.DataFrame()

patient_df = load_patient_data()

# SESSION STATE
if "login" not in st.session_state:
    st.session_state.login = False

# LOGIN PAGE
def doctor_login():
    st.title("👨‍⚕️ Doctor Login Portal")
    st.markdown("### Secure Hospital Access")

    col1, col2 = st.columns(2)
    with col1:
        user = st.text_input("Doctor ID")
    with col2:
        pwd = st.text_input("Password", type="password")

    if st.button("🔐 Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.login = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Doctor ID or Password")

    st.info("Demo Login → admin / admin123")

# DASHBOARD
def risk_prediction():
    st.title("🩺 Patient Risk Prediction Dashboard")

    #SIDEBAR PATIENT SELECTION
    patient_id = None
    if not patient_df.empty:
        st.sidebar.header("Select Patient")
        patient_id = st.sidebar.selectbox("Patient ID", patient_df["patient_id"].tolist())

    # SIDEBAR IMAGE UPLOAD
    st.sidebar.header("📸 Patient Image Upload")
    uploaded_image = st.sidebar.file_uploader(
        "Upload Patient Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.sidebar.image(image, caption="Patient Image", use_container_width=True)

    # AUTO-FILL PATIENT DATA
    patient_row = None
    if patient_id:
        patient_row = patient_df[patient_df["patient_id"] == patient_id].iloc[0]

    st.markdown("### Enter Patient Medical Details")

    # NUMERIC INPUTS
    c1, c2, c3 = st.columns(3)
    with c1:
        age = safe_number_input("Age", 0, 120, int(patient_row["age"]) if patient_row is not None else 45)
        sys = safe_number_input("Systolic BP", 80, 200, int(patient_row["blood_pressure_systolic"]) if patient_row is not None else 120)
        dia = safe_number_input("Diastolic BP", 50, 130, int(patient_row["blood_pressure_diastolic"]) if patient_row is not None else 80)
        heart = safe_number_input("Heart Rate", 40, 200, int(patient_row["heart_rate"]) if patient_row is not None else 72)
        bmi = safe_number_input("BMI", 10.0, 60.0, float(patient_row["bmi"]) if patient_row is not None else 24.0, step=0.1)

    with c2:
        sugar = safe_number_input("Blood Sugar", 50, 400, int(patient_row["blood_sugar"]) if patient_row is not None else 110)
        chol = safe_number_input("Cholesterol", 100, 400, int(patient_row["cholesterol"]) if patient_row is not None else 180)
        oxy = safe_number_input("Oxygen Saturation", 70, 100, int(patient_row["oxygen_saturation"]) if patient_row is not None else 98)
        resp = safe_number_input("Respiratory Rate", 10, 40, int(patient_row["respiratory_rate"]) if patient_row is not None else 18)
        temp = safe_number_input("Body Temperature", 34.0, 42.0, float(patient_row["body_temperature"]) if patient_row is not None else 36.8, step=0.1)

    with c3:
        severity = safe_number_input("Severity Score", 0, 10, int(patient_row["severity_score"]) if patient_row is not None else 3)
        meds = safe_number_input("Medication Count", 0, 20, int(patient_row["medication_count"]) if patient_row is not None else 2)
        stay = safe_number_input("Length of Stay", 0, 60, int(patient_row["length_of_stay"]) if patient_row is not None else 3)
        cost = safe_number_input("Treatment Cost", 0, 1_000_000, int(patient_row["treatment_cost"]) if patient_row is not None else 50000)

    diabetes = safe_selectbox("Diabetes", [0, 1], int(patient_row["diabetes"]) if patient_row is not None else 0)
    heart_disease = safe_selectbox("Heart Disease", [0, 1], int(patient_row["heart_disease"]) if patient_row is not None else 0)

    # CATEGORICAL INPUTS
    c4, c5, c6 = st.columns(3)
    with c4:
        gender = safe_selectbox("Gender", ["Male", "Female"], patient_row["gender"] if patient_row is not None else "Male")
        blood = safe_selectbox("Blood Type", ["A", "B", "AB", "O"], patient_row["blood_type"] if patient_row is not None else "A")
        dept = safe_selectbox("Department", ["Cardiology", "Neurology", "Orthopedics", "General"], patient_row["department"] if patient_row is not None else "Cardiology")

    with c5:
        admission = safe_selectbox("Admission Type", ["Emergency", "Planned"], patient_row["admission_type"] if patient_row is not None else "Emergency")
        insurance = safe_selectbox("Insurance", ["Yes", "No"], patient_row["insurance"] if patient_row is not None else "Yes")
        recovery = safe_selectbox("Recovery Status", ["Recovered", "Critical"], patient_row["recovery_status"] if patient_row is not None else "Recovered")

    with c6:
        smoking = safe_selectbox("Smoking Status", ["Never", "Former", "Current"], patient_row["smoking_status"] if patient_row is not None else "Never")
        alcohol = safe_selectbox("Alcohol Consumption", ["Yes", "No"], patient_row["alcohol_consumption"] if patient_row is not None else "No")
        outcome = safe_selectbox("Outcome", ["Improved", "Worsened"], patient_row["outcome"] if patient_row is not None else "Improved")
        icu = safe_selectbox("ICU", ["Yes", "No"], patient_row["ICU"] if patient_row is not None else "No")

    # PREDICTION 
    if st.button("🔍 Predict Patient Risk"):
        patient_data = {
            "age": age,
            "blood_pressure_systolic": sys,
            "blood_pressure_diastolic": dia,
            "heart_rate": heart,
            "blood_sugar": sugar,
            "diabetes": diabetes,
            "heart_disease": heart_disease,
            "cholesterol": chol,
            "oxygen_saturation": oxy,
            "respiratory_rate": resp,
            "body_temperature": temp,
            "bmi": bmi,
            "medication_count": meds,
            "length_of_stay": stay,
            "treatment_cost": cost,
            "severity_score": severity,
            "gender": gender,
            "blood_type": blood,
            "department": dept,
            "admission_type": admission,
            "insurance": insurance,
            "recovery_status": recovery,
            "smoking_status": smoking,
            "alcohol_consumption": alcohol,
            "outcome": outcome,
            "ICU": icu,
            "outcome_label": 0
        }

        df = pd.DataFrame([patient_data])
        df = df[expected_features]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        if pred == 1:
            st.error(f"⚠️ HIGH RISK PATIENT\n\nRisk Probability: {prob:.2%}")
        else:
            st.success(f"✅ LOW RISK PATIENT\n\nRisk Probability: {prob:.2%}")

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.login = False
        st.rerun()

# APP FLOW 
if not st.session_state.login:
    doctor_login()
else:
    risk_prediction()
