import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"  # No /book

# Streamlit UI
st.title("📅 Women's Healthcare Appointment Booking")

# Input fields
user_name = st.text_input("👩 Your Name")
doctor_id = st.number_input("🔢 Doctor ID", min_value=1, step=1)
date = st.date_input("📆 Select Date")
time = st.time_input("⏰ Select Time")
contact = st.text_input("📞 Contact Number", max_chars=15)
email = st.text_input("📧 Email Address")

# Submit Button
if st.button("Book Appointment"):
    appointment_data = {
        "user_name": user_name,
        "doctor_id": doctor_id,
        "date": str(date),
        "time": str(time),
        "contact": contact,
        "email": email
    }

    # Sending POST request to FastAPI
    response = requests.post(f"{FASTAPI_URL}/book", json=appointment_data)

    # Handling response
    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ {result['message']}")
        st.write(f"**Doctor:** {result['doctor']['name']} ({result['doctor']['specialization']})")
        st.write(f"**Location:** {result['doctor']['location']}")
        st.write(f"📅 **Date:** {result['date']} ⏰ **Time:** {result['time']}")
    else:
        st.error("❌ Failed to book appointment. Please check details.")
