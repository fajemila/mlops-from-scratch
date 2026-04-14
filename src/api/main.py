from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

app = FastAPI(title="AQI Prediction API")

# 1. Define the incoming data structure
class WeatherInput(BaseModel):
    dew_point: float
    temp: float
    pressure: float
    wind_speed: float
    snow: float
    rain: float
    wind_dir_NE: bool
    wind_dir_NW: bool
    wind_dir_SE: bool

# 2. Load the latest model from MLflow (assuming run ID is known or latest is fetched)
# For this example, we point to the local mlruns folder where MLflow saves the default model
MODEL_PATH = "mlruns/0/model" # In a real setup, you'd query MLflow for the exact Run ID

try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
except Exception as e:
    print(f"Model not found. Make sure to run the training script first! Error: {e}")
    model = None

@app.get("/")
def health_check():
    return {"status": "API is running!"}

@app.post("/predict")
def predict_aqi(data: WeatherInput):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convert incoming JSON data into a pandas DataFrame for the model
    input_df = pd.DataFrame([data.model_dump()])
    
    # Make the prediction
    prediction = model.predict(input_df)
    
    return {"predicted_pm25": float(prediction[0])}