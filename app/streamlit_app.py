import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Flight Price Predictor")
st.markdown("Предсказание стоимости авиабилета на основе характеристик рейса")
st.divider()

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Авиакомпания",
        ["Vistara", "Air_India", "Indigo", "SpiceJet", "AirAsia", "GO_FIRST"],
    )
    source_city = st.selectbox(
        "Город отправления",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
    )
    destination_city = st.selectbox(
        "Город назначения",
        ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
    )
    flight_class = st.radio("Класс", ["Economy", "Business"], horizontal=True)

with col2:
    departure_time = st.selectbox(
        "Время вылета",
        ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
        index=1,
    )
    arrival_time = st.selectbox(
        "Время прилёта",
        ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
        index=2,
    )
    stops = st.selectbox("Пересадки", ["zero", "one", "two_or_more"])
    duration = st.slider("Длительность полёта (часы)", 0.5, 50.0, 2.5, 0.5)
    days_left = st.slider("Дней до вылета", 1, 49, 20)

st.divider()

if st.button("🔮 Предсказать цену", use_container_width=True, type="primary"):
    if source_city == destination_city:
        st.error("Город отправления и назначения не могут совпадать!")
    else:
        payload = {
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "stops": stops,
            "flight_class": flight_class,
            "duration": duration,
            "days_left": days_left,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                price = result["predicted_price"]
                st.success(f"### 💰 Предсказанная цена: ₹ {price:,.0f}")
                st.caption(f"Модель: {result['model']} | Валюта: {result['currency']}")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Маршрут", f"{source_city} → {destination_city}")
                col_b.metric("Класс", flight_class)
                col_c.metric("Дней до вылета", days_left)
            else:
                st.error(f"Ошибка API: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API. Убедитесь что FastAPI запущен на порту 8000.")

st.divider()
st.caption("Данные: Flight Price Prediction Dataset (Kaggle) | Модель: LightGBM + Optuna")
