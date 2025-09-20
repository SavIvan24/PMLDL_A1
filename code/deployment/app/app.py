import streamlit as st
import requests
import json

# Title of the app
st.title("🍷 Wine Classifier App")

# Description
st.write("""
This app predicts the **Wine Class** based on its chemical properties!
Fill in the features below and click 'Predict'.
""")

# Create input fields for each feature
# (Using a subset of features for simplicity in the UI)
alcohol = st.number_input('Alcohol', min_value=10.0, max_value=15.0, value=13.0)
malic_acid = st.number_input('Malic Acid', min_value=0.0, max_value=5.0, value=2.0)
ash = st.number_input('Ash', min_value=1.0, max_value=3.0, value=2.3)
proline = st.number_input('Proline', min_value=100.0, max_value=2000.0, value=1000.0)

# Button to make prediction
if st.button('Predict'):
    # Prepare the data to send to the API
    input_data = {
        "alcohol": alcohol,
        "malic_acid": malic_acid,
        "ash": ash,
        "alcalinity_of_ash": 25.0,  # Example fixed value for demo
        "magnesium": 100.0,         # Example fixed value for demo
        "total_phenols": 2.0,       # Example fixed value for demo
        "flavanoids": 2.0,          # Example fixed value for demo
        "nonflavanoid_phenols": 0.3,# Example fixed value for demo
        "proanthocyanins": 1.5,     # Example fixed value for demo
        "color_intensity": 5.0,     # Example fixed value for demo
        "hue": 1.0,                 # Example fixed value for demo
        "od280_od315_of_diluted_wines": 3.0, # Example fixed value for demo
        "proline": proline
    }

    # Define the API endpoint URL
    # 'api' is the service name defined in docker-compose.yml
    api_url = "http://api:8000/predict"

    try:
        # Send POST request to the API
        response = requests.post(api_url, json=input_data)
        prediction = response.json()

        # Display the prediction
        st.success(f"Predicted Wine Class: **{prediction['prediction']}**")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")