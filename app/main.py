from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Flight Price Prediction API",
    description="Предсказание цены авиабилета по характеристикам рейса",
    version="1.0.0",
)

# Загружаем модель и признаки
MODEL_PATH = Path(__file__).parent.parent / "models" / "final_model.pkl"
FEATURES_PATH = Path(__file__).parent.parent / "models" / "features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# Маппинги для кодирования
AIRLINE_MAP = {
    "SpiceJet": 0, "AirAsia": 1, "Vistara": 2,
    "GO_FIRST": 3, "Indigo": 4, "Air_India": 5,
}
CITY_MAP = {
    "Delhi": 0, "Mumbai": 1, "Bangalore": 2,
    "Kolkata": 3, "Hyderabad": 4, "Chennai": 5,
}
TIME_MAP = {
    "Early_Morning": 0, "Morning": 1, "Afternoon": 2,
    "Evening": 3, "Night": 4, "Late_Night": 5,
}
STOPS_MAP = {"zero": 0, "one": 1, "two_or_more": 2}
CLASS_MAP = {"Economy": 0, "Business": 1}
BOOKING_MAP = {
    "long_advance": 0, "medium_advance": 1,
    "short_advance": 2, "last_minute": 3,
}


class FlightInput(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str
    flight_class: str
    duration: float
    days_left: int

    class Config:
        json_schema_extra = {
            "example": {
                "airline": "Vistara",
                "source_city": "Delhi",
                "destination_city": "Mumbai",
                "departure_time": "Morning",
                "arrival_time": "Afternoon",
                "stops": "zero",
                "flight_class": "Economy",
                "duration": 2.17,
                "days_left": 30,
            }
        }


class PredictionOutput(BaseModel):
    predicted_price: float
    currency: str = "INR"
    model: str = "LightGBM (Optuna)"


def encode_features(data: FlightInput) -> pd.DataFrame:
    route = f"{data.source_city}_{data.destination_city}"
    route_map = {
        "Delhi_Mumbai": 0, "Mumbai_Delhi": 1,
        "Delhi_Bangalore": 2, "Bangalore_Delhi": 3,
        "Delhi_Kolkata": 4, "Kolkata_Delhi": 5,
        "Delhi_Hyderabad": 6, "Hyderabad_Delhi": 7,
        "Delhi_Chennai": 8, "Chennai_Delhi": 9,
        "Mumbai_Bangalore": 10, "Bangalore_Mumbai": 11,
    }

    is_last_minute = int(data.days_left <= 7)
    if data.days_left <= 7:
        booking_cat = "last_minute"
    elif data.days_left <= 30:
        booking_cat = "short_advance"
    elif data.days_left <= 60:
        booking_cat = "medium_advance"
    else:
        booking_cat = "long_advance"

    row = {
        "duration": data.duration,
        "days_left": data.days_left,
        "duration_minutes": data.duration * 60,
        "is_direct": int(data.stops == "zero"),
        "is_business": int(data.flight_class == "Business"),
        "is_last_minute": is_last_minute,
        "is_long_flight": int(data.duration > 5.0),
        "stops_encoded": STOPS_MAP.get(data.stops, 1),
        "departure_time_encoded": TIME_MAP.get(data.departure_time, 2),
        "arrival_time_encoded": TIME_MAP.get(data.arrival_time, 2),
        "booking_category_encoded": BOOKING_MAP.get(booking_cat, 1),
        "airline_encoded": AIRLINE_MAP.get(data.airline, 0),
        "source_city_encoded": CITY_MAP.get(data.source_city, 0),
        "destination_city_encoded": CITY_MAP.get(data.destination_city, 1),
        "route_encoded": route_map.get(route, 0),
        "class_encoded": CLASS_MAP.get(data.flight_class, 0),
    }
    return pd.DataFrame([row])


@app.get("/")
def root():
    return {"message": "Flight Price Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: FlightInput):
    df = encode_features(data)
    df = df[features]
    price = float(model.predict(df)[0])
    return PredictionOutput(predicted_price=round(price, 2))
