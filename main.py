from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("air_quality_model.pkl")

import os
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

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

@app.post("/ai-advice")
async def ai_advice(data: SensorData):
    quality = "Good" if data.gas < 1000 else "Moderate" if data.gas < 2000 else "Bad"

    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{
                    "role": "user",
                    "content": f"You are an indoor air quality expert. Based on these IoT sensor readings, give 3 short specific actionable recommendations in bullet points. Be concise.\n\nTemperature: {data.temperature}°C\nHumidity: {data.humidity}%\nGas Level: {data.gas}\nAir Quality: {quality}\n\nGive exactly 3 bullet points, each under 15 words."
                }]
            },
            timeout=30.0
        )
        result = res.json()
        return {"advice": result["content"][0]["text"]}
