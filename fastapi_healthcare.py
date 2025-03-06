from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date, time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pickle
import numpy as np

app = FastAPI()

# --- Load healthcare data ---
try:
    healthcare_data = pd.read_csv("processed_healthcare_data.csv")
except FileNotFoundError:
    raise FileNotFoundError("processed_healthcare_data.csv not found.  Make sure it's in the same directory.")

# --- Data Preprocessing ---
# Identify categorical and numerical columns
categorical_cols = healthcare_data.select_dtypes(include='object').columns
numerical_cols = healthcare_data.select_dtypes(include=np.number).columns

# Initialize LabelEncoders and Scaler
label_encoders = {}
scaler = StandardScaler()

# Fit LabelEncoders for each categorical column
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    healthcare_data[col] = label_encoders[col].fit_transform(healthcare_data[col])

# Scale numerical features
healthcare_data[numerical_cols] = scaler.fit_transform(healthcare_data[numerical_cols])

# --- Clustering Model Training ---
# Select features for clustering (ensure consistency)
features = ['Age', 'Gender', 'Location', 'Existing_Medical_Conditions', 'Past_Surgeries', 'Family_History_of_Diseases', 'Allergies', 'Medications_Currently_Taking', 'Dietary_Preferences', 'Exercise_Routine', 'Sleep_Patterns', 'Stress_Levels', 'PCOS_Diagnosed', 'Menstrual_Cycle_Irregular', 'Menstrual_Flow_Type', 'Facial_Hair_Growth', 'Acne_Severity', 'Weight_Gain_Issues', 'Hormonal_Levels_Elevated', 'Pregnancy_Status', 'High_Risk_Pregnancy', 'Previous_Pregnancy_Complications', 'Gestational_Diabetes', 'Prenatal_Vitamin_Intake', 'Postpartum_Depression_Symptoms']

X = healthcare_data[features]

# Train Hierarchical Clustering model
n_clusters = 5  # Adjust as needed
hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
healthcare_data['Cluster_HC'] = hc_model.fit_predict(X)

# --- Save preprocessing objects and the clustering model ---
pickle.dump(label_encoders, open("label_encoders.pkl", 'wb'))
pickle.dump(scaler, open("scaler.pkl", 'wb'))
pickle.dump(hc_model, open("hierarchical_clustering_model.pkl", 'wb'))

# --- Load the clustering model and preprocessing objects ---
label_encoders = pickle.load(open("label_encoders.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))
loaded_model = pickle.load(open("hierarchical_clustering_model.pkl", 'rb'))

# Sample doctor data (replace with a database in production)
try:
    doctors_df = pd.read_csv("doctors.csv")
    doctors = doctors_df.to_dict(orient='records')
except FileNotFoundError:
    doctors = [
        {"id": 1, "name": "Dr. A", "specialization": "Cardiologist", "location": "Mumbai", "cluster": 0},
        {"id": 2, "name": "Dr. B", "specialization": "Dermatologist", "location": "Delhi", "cluster": 1},
        {"id": 3, "name": "Dr. C", "specialization": "Gynecologist", "location": "Mumbai", "cluster": 0},
        {"id": 4, "name": "Dr. D", "specialization": "Neurologist", "location": "Delhi", "cluster": 1},
        {"id": 5, "name": "Dr. E", "specialization": "Cardiologist", "location": "Bangalore", "cluster": 2},
        {"id": 6, "name": "Dr. F", "specialization": "Dermatologist", "location": "Kolkata", "cluster": 3},
        {"id": 7, "name": "Dr. G", "specialization": "Gynecologist", "location": "Chennai", "cluster": 4},
        {"id": 8, "name": "Dr. H", "specialization": "Neurologist", "location": "Pune", "cluster": 2},
    ]

class UserData(BaseModel):
    Age: int
    Gender: int  # Use numerical values (0 or 1)
    Location: int  # Use numerical values (0, 1, 2...)
    Existing_Medical_Conditions: str
    Past_Surgeries: str
    Family_History_of_Diseases: str
    Allergies: str
    Medications_Currently_Taking: str
    Dietary_Preferences: str
    Exercise_Routine: str
    Sleep_Patterns: str
    Stress_Levels: str
    PCOS_Diagnosed: str
    Menstrual_Cycle_Irregular: str
    Menstrual_Flow_Type: str
    Facial_Hair_Growth: str
    Acne_Severity: str
    Weight_Gain_Issues: str
    Hormonal_Levels_Elevated: str
    Pregnancy_Status: str
    High_Risk_Pregnancy: str
    Previous_Pregnancy_Complications: str
    Gestational_Diabetes: str
    Prenatal_Vitamin_Intake: str
    Postpartum_Depression_Symptoms: str

class AppointmentRequest(BaseModel):
    user_name: str
    doctor_id: int
    date: date
    time: str  # Keeping it as string for simplicity
    contact: str
    email: str

@app.post("/predict_cluster")
def predict_cluster(user_data: UserData):
    """Predicts the cluster for a given user based on their health data."""
    user_df = pd.DataFrame([user_data.dict()])  # Create DataFrame with index

    # Preprocess user data using the fitted LabelEncoders
    for col in categorical_cols:
        # Check if the value exists in the encoder's classes
        if user_data.dict()[col] in label_encoders[col].classes_:
            user_df[col] = label_encoders[col].transform([user_data.dict()[col]])
        else:
            # Handle unseen values (e.g., assign a default value)
            user_df[col] = -1  # Or another appropriate default value
    # Scale numerical features
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

    # Ensure all columns are present in user_df
    for feature in features:
        if feature not in user_df.columns:
            user_df[feature] = 0  # Fill missing columns with 0

    user_cluster = loaded_model.predict(user_df[features])[0]  # Predict the cluster using the loaded model
    return {"cluster": int(user_cluster)}

@app.get("/doctors/{cluster}")
def get_doctors_by_cluster(cluster: int, location: str = None, specialization: str = None):
    """Returns a list of doctors for a specific cluster, optionally filtered by location and specialization."""
    recommended_doctors = [
        doctor
        for doctor in doctors
        if doctor["cluster"] == cluster and
           (location is None or doctor["location"] == doctor["location"]) and
           (specialization is None or doctor["specialization"] == doctor["specialization"])
    ]
    if not recommended_doctors:
        raise HTTPException(status_code=404, detail="No doctors found for this cluster and criteria.")
    return recommended_doctors

@app.post("/book")
def book_appointment(data: AppointmentRequest):
    """Books an appointment for a user with a specific doctor."""
    if data.doctor_id not in [doctor['id'] for doctor in doctors]:
        raise HTTPException(status_code=404, detail="Doctor not found")

    appointment_details = {
        "message": "Appointment booked successfully!",
        "doctor": next(doctor for doctor in doctors if doctor['id'] == data.doctor_id),
        "date": data.date,
        "time": data.time,
    }

    return appointment_details

