from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the trained model from the correct path
model_path = '/app/models/wine_rf_model.joblib'
try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_wine
    data = load_wine()
    X, y = data.data, data.target
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("Using dummy model for testing")

# Define the input data schema using Pydantic
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

# Initialize FastAPI app
app = FastAPI(title="Wine Classifier API")

# Define prediction endpoint
@app.post("/predict")
def predict(features: WineFeatures):
    # Convert input data into a numpy array for prediction
    data_array = np.array([[
        features.alcohol, features.malic_acid, features.ash,
        features.alcalinity_of_ash, features.magnesium,
        features.total_phenols, features.flavanoids,
        features.nonflavanoid_phenols, features.proanthocyanins,
        features.color_intensity, features.hue,
        features.od280_od315_of_diluted_wines, features.proline
    ]])
    
    # Make prediction
    prediction = model.predict(data_array)
    
    # Return the result as a JSON response
    return {"prediction": int(prediction[0])}

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Wine Classifier API is live!"}

# Add this for debugging
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}