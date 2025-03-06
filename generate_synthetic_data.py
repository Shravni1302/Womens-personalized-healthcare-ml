import pandas as pd
import numpy as np
from faker import Faker
from geopy.geocoders import Nominatim
import random

fake = Faker()
geolocator = Nominatim(user_agent="geoapi")

# --- Configuration ---
num_samples = 500
locations = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"]

# Probabilities based on real world dataset
health_issue_probabilities = {
    "PCOS": 0.3,
    "Maternal Health": 0.25,
    "General Health": 0.45
}

symptoms_probabilities = {
    "Irregular periods": 0.3,
    "Weight gain": 0.2,
    "Pregnancy concerns": 0.25,
    "Other": 0.25
}

# --- Function to generate random data with probabilities ---
def generate_data():
    location = np.random.choice(locations)
    age = np.random.randint(18, 50)

    # Generate health issue based on probabilities
    health_issue = np.random.choice(list(health_issue_probabilities.keys()), p=list(health_issue_probabilities.values()))

    # Generate symptoms based on probabilities
    symptoms = np.random.choice(list(symptoms_probabilities.keys()), p=list(symptoms_probabilities.values()))

    return [fake.name(), age, location, health_issue, symptoms]

# --- Generate data ---
data = [generate_data() for _ in range(num_samples)]
df = pd.DataFrame(data, columns=["Name", "Age", "Location", "Health_Issue", "Symptoms"])

# --- Create additional columns with distributions and correlations ---

# Gender distribution (simulating a slightly higher proportion of females)
df['Gender'] = np.random.choice(['Female', 'Male'], size=num_samples, p=[0.6, 0.4])

# Simulate PCOS based on age (higher probability for women in their 20s and 30s)
df['PCOS_Diagnosed'] = df.apply(lambda row: 'Yes' if (row['Age'] >= 20 and row['Age'] <= 39 and random.random() < 0.3) else 'No', axis=1)

# Simulate Menstrual Cycle Irregularities (correlated with PCOS)
df['Menstrual_Cycle_Irregular'] = df.apply(lambda row: 'Yes' if row['PCOS_Diagnosed'] == 'Yes' and random.random() < 0.7 else 'No', axis=1)

# Simulate a correlation between age and existing medical conditions
def get_medical_condition(age):
    if age > 40:
        conditions = ['Diabetes', 'Hypertension', 'Thyroid Disorder', 'None']
        probabilities = [0.2, 0.2, 0.1, 0.5]  # Higher chance of having a condition
    else:
        conditions = ['PCOS', 'None']
        probabilities = [0.15, 0.85]  # Lower chance of having a condition
    return np.random.choice(conditions, p=probabilities)

df['Existing_Medical_Conditions'] = df['Age'].apply(get_medical_condition)

# Introduce some missing values (e.g., 10% missing in 'Past_Surgeries')
df['Past_Surgeries'] = np.random.choice(['Yes', 'No', None], size=num_samples, p=[0.1, 0.8, 0.1])

# Fill missing values with 'Unknown'
df = df.fillna('Unknown')

# Convert all column to string type
df = df.astype(str)

# Save to CSV
df.to_csv("synthetic_healthcare_data_comprehensive.csv", index=False)
