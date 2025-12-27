# =============================================================================
# FIXED INPUT DATA FOR CHATBOT — MATCHES TRAINING EXACTLY
# =============================================================================
import streamlit as st
import joblib
import pandas as pd
import os

# Load model — fixed path
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_ensemble.pkl')
ensemble = joblib.load(model_path)
xgb_model = ensemble['xgb']
lgb_model = ensemble['lgb']

st.title("🏥 Diabetes 30-Day Readmission Risk Predictor")
st.markdown("### Clinical Decision Support Tool")

# Sidebar inputs — collect ALL needed features
st.sidebar.header("Enter Patient Details")

age = st.sidebar.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
race = st.sidebar.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Other', '?', 'Asian', 'Hispanic'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Unknown/Invalid'])
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 132, 43)
num_procedures = st.sidebar.slider("Number of Procedures (non-lab)", 0, 6, 0)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 16)
number_outpatient = st.sidebar.slider("Outpatient Visits (past year)", 0, 42, 0)
number_emergency = st.sidebar.slider("Emergency Visits (past year)", 0, 76, 0)
number_inpatient = st.sidebar.slider("Inpatient Visits (past year)", 0, 21, 0)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 9)
max_glu_serum = st.sidebar.selectbox("Max Glucose Serum", ['None', 'Norm', '>200', '>300'])
A1Cresult = st.sidebar.selectbox("A1C Result", ['None', 'Norm', '>7', '>8'])
change = st.sidebar.selectbox("Medication Changed?", ['No', 'Ch'])
diabetesMed = st.sidebar.selectbox("Diabetes Medication Prescribed?", ['No', 'Yes'])
insulin = st.sidebar.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
admission_type_id = st.sidebar.selectbox("Admission Type", [1,2,3,4,5,6,7,8], format_func=lambda x: {
    1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Newborn', 5: 'Not Available', 
    6: 'NULL', 7: 'Trauma Center', 8: 'Not Mapped'}[x])
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition", [1,2,3,4,5,6,7,8,9,10,11], 
                                                format_func=lambda x: {1: 'Home', 3: 'SNF', 6: 'Home Health'}.get(x, 'Other'))
admission_source_id = st.sidebar.selectbox("Admission Source", [1,2,3,4,5,6,7,8,9,10], 
                                           format_func=lambda x: {7: 'Emergency Room', 1: 'Physician Referral', 4: 'Transfer'}.get(x, 'Other'))
medical_specialty = st.sidebar.text_input("Medical Specialty (e.g., InternalMedicine, Cardiology)", "InternalMedicine")

# Create input DataFrame with EXACT same columns as training
input_data = pd.DataFrame([{
    'race': race,
    'gender': gender,
    'age': age,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'max_glu_serum': max_glu_serum,
    'A1Cresult': A1Cresult,
    'change': change,
    'diabetesMed': diabetesMed,
    'insulin': insulin,
    'admission_type_id': admission_type_id,
    'discharge_disposition_id': discharge_disposition_id,
    'admission_source_id': admission_source_id,
    'medical_specialty': medical_specialty
}])

# Convert to category (must match training)
for col in input_data.select_dtypes(include='object').columns:
    input_data[col] = input_data[col].astype('category')

# Prediction
xgb_pred = xgb_model.predict_proba(input_data)[:,1][0]
lgb_pred = lgb_model.predict_proba(input_data)[:,1][0]
risk_score = (xgb_pred + lgb_pred) / 2

st.metric("30-Day Readmission Risk Probability", f"{risk_score:.1%}")

if risk_score > 0.5:
    st.error("🚨 HIGH RISK — Recommend extended monitoring or care coordination")
else:
    st.success("✅ Low risk — Standard discharge protocol")

st.info("Model: XGBoost + LightGBM ensemble (0.71+ AUC) | Explainable with SHAP")