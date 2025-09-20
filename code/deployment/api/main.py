from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load the trained model from the correct path
model_path = '/app/models/wine_rf_model.joblib'
model = joblib.load(model_path)

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