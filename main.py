from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("air_quality_model.pkl")

class SensorData(BaseModel):
    temperature: float
    humidity: float
    gas: int

@app.get("/")
def root():
    return {"status": "Air Monitor ML API is running!"}

@app.post("/predict")
def predict(data: SensorData):
    features = np.array([[data.temperature, data.humidity, data.gas]])
    predicted_gas = model.predict(features)[0]

    if predicted_gas < 1000:
        quality = "Good"
        advice = "Air quality is great! Keep windows closed to maintain it."
    elif predicted_gas < 2000:
        quality = "Moderate"
        advice = "Consider opening a window for fresh air."
    else:
        quality = "Bad"
        advice = "Open windows immediately and turn on a fan!"

    return {
        "current_gas": data.gas,
        "predicted_gas": round(predicted_gas),
        "predicted_quality": quality,
        "advice": advice,
        "temperature": data.temperature,
        "humidity": data.humidity
    }
