import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"

st.title("🔍 Personalized Healthcare Recommendation")

# --- User Data Input Form ---
st.header("Patient Information")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=[0, 1])  # Assuming 0=Female, 1=Male
location = st.selectbox("Location", options=[0, 1, 2, 3])  # Assuming numerical location values
existing_medical_conditions = st.text_input("Existing Medical Conditions")
past_surgeries = st.text_input("Past Surgeries")
family_history_of_diseases = st.text_input("Family History of Diseases")
allergies = st.text_input("Allergies")
medications_currently_taking = st.text_input("Medications Currently Taking")
dietary_preferences = st.text_input("Dietary Preferences")
exercise_routine = st.text_input("Exercise Routine")
sleep_patterns = st.text_input("Sleep Patterns")
stress_levels = st.text_input("Stress Levels")
pcos_diagnosed = st.text_input("PCOS Diagnosed")
menstrual_cycle_irregular = st.text_input("Menstrual Cycle Irregular")
menstrual_flow_type = st.text_input("Menstrual Flow Type")
facial_hair_growth = st.text_input("Facial Hair Growth")
acne_severity = st.text_input("Acne Severity")
weight_gain_issues = st.text_input("Weight Gain Issues")
hormonal_levels_elevated = st.text_input("Hormonal Levels Elevated")
pregnancy_status = st.text_input("Pregnancy Status")
high_risk_pregnancy = st.text_input("High Risk Pregnancy")
previous_pregnancy_complications = st.text_input("Previous Pregnancy Complications")
gestational_diabetes = st.text_input("Gestational Diabetes")
prenatal_vitamin_intake = st.text_input("Prenatal Vitamin Intake")
postpartum_depression_symptoms = st.text_input("Postpartum Depression Symptoms")

# --- Doctor Search Form ---
st.header("Doctor Search")
location_filter = st.selectbox("Select Location:", ["Mumbai", "Delhi", "Bangalore", "Kolkata"])
specialization_filter = st.selectbox("Select Specialization:", ["Cardiologist", "Dermatologist", "Gynecologist", "Neurologist"])

if st.button("Find Doctors"):
    # Create user data dictionary
    user_data = {
        "Age": age,
        "Gender": gender,
        "Location": location,
        "Existing_Medical_Conditions": existing_medical_conditions,
        "Past_Surgeries": past_surgeries,
        "Family_History_of_Diseases": family_history_of_diseases,
        "Allergies": allergies,
        "Medications_Currently_Taking": medications_currently_taking,
        "Dietary_Preferences": dietary_preferences,
        "Exercise_Routine": exercise_routine,
        "Sleep_Patterns": sleep_patterns,
        "Stress_Levels": stress_levels,
        "PCOS_Diagnosed": pcos_diagnosed,
        "Menstrual_Cycle_Irregular": menstrual_cycle_irregular,
        "Menstrual_Flow_Type": menstrual_flow_type,
        "Facial_Hair_Growth": facial_hair_growth,
        "Acne_Severity": acne_severity,
        "Weight_Gain_Issues": weight_gain_issues,
        "Hormonal_Levels_Elevated": hormonal_levels_elevated,
        "Pregnancy_Status": pregnancy_status,
        "High_Risk_Pregnancy": high_risk_pregnancy,
        "Previous_Pregnancy_Complications": previous_pregnancy_complications,
        "Gestational_Diabetes": gestational_diabetes,
        "Prenatal_Vitamin_Intake": prenatal_vitamin_intake,
        "Postpartum_Depression_Symptoms": postpartum_depression_symptoms
    }

    # --- Get Cluster Prediction ---
    cluster_response = requests.post(f"{API_URL}/predict_cluster", json=user_data)
    if cluster_response.status_code == 200:
        cluster_data = cluster_response.json()
        user_cluster = cluster_data["cluster"]
        st.write(f"Predicted Cluster: {user_cluster}")

        # --- Fetch Doctors by Cluster, Location, and Specialization ---
        doctors_response = requests.get(f"{API_URL}/doctors/{user_cluster}", params={"location": location_filter, "specialization": specialization_filter})
        if doctors_response.status_code == 200:
            doctors = doctors_response.json()
            if doctors:
                df = pd.DataFrame(doctors)
                st.dataframe(df)
            else:
                st.warning("No doctors found for your cluster and criteria.")
        else:
            st.error(f"Failed to fetch doctors: {doctors_response.text}")
    else:
        st.error(f"Failed to predict cluster: {cluster_response.text}")
