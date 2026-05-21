"""
Скрипт для обучения финальной модели и сохранения в models/final_model.pkl
Запуск: python src/train_model.py
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "Clean_Dataset.csv"
PROCESSED_PATH = ROOT / "data" / "processed"
MODELS_PATH = ROOT / "models"

PROCESSED_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

print("1. Загружаем данные...")
df = pd.read_csv(DATA_PATH)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
print(f"   Загружено: {df.shape[0]:,} строк, {df.shape[1]} колонок")

print("2. Очистка...")
df = df.drop_duplicates()
Q1 = df['price'].quantile(0.01)
Q3 = df['price'].quantile(0.99)
df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
print(f"   После очистки: {df.shape[0]:,} строк")

print("3. Feature Engineering...")
df['route'] = df['source_city'] + '_' + df['destination_city']
df['is_direct'] = (df['stops'] == 'zero').astype(int)
df['is_business'] = (df['class'] == 'Business').astype(int)
df['is_last_minute'] = (df['days_left'] <= 7).astype(int)
df['duration_minutes'] = df['duration'] * 60
df['is_long_flight'] = (df['duration'] > df['duration'].median()).astype(int)

def booking_category(days):
    if days <= 7:
        return 'last_minute'
    elif days <= 30:
        return 'short_advance'
    elif days <= 60:
        return 'medium_advance'
    else:
        return 'long_advance'

df['booking_category'] = df['days_left'].apply(booking_category)

print("4. Кодирование признаков...")
stops_order = [['zero', 'one', 'two_or_more']]
oe_stops = OrdinalEncoder(categories=stops_order)
df['stops_encoded'] = oe_stops.fit_transform(df[['stops']])

time_order = [['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']]
oe_time = OrdinalEncoder(categories=time_order, handle_unknown='use_encoded_value', unknown_value=-1)
df['departure_time_encoded'] = oe_time.fit_transform(df[['departure_time']])
df['arrival_time_encoded'] = oe_time.fit_transform(df[['arrival_time']])

booking_order = [['long_advance', 'medium_advance', 'short_advance', 'last_minute']]
oe_booking = OrdinalEncoder(categories=booking_order)
df['booking_category_encoded'] = oe_booking.fit_transform(df[['booking_category']])

le = LabelEncoder()
for col in ['airline', 'source_city', 'destination_city', 'route', 'class', 'flight']:
    df[col + '_encoded'] = le.fit_transform(df[col])

FEATURES = [
    'duration', 'days_left', 'duration_minutes',
    'is_direct', 'is_business', 'is_last_minute', 'is_long_flight',
    'stops_encoded', 'departure_time_encoded', 'arrival_time_encoded',
    'booking_category_encoded', 'airline_encoded', 'source_city_encoded',
    'destination_city_encoded', 'route_encoded', 'class_encoded',
]
TARGET = 'price'

print("5. Сплит данных...")
X = df[FEATURES]
y = df[TARGET]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=SEED)

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Сохраняем сплиты
X_train.to_csv(PROCESSED_PATH / 'X_train.csv', index=False)
X_val.to_csv(PROCESSED_PATH / 'X_val.csv', index=False)
X_test.to_csv(PROCESSED_PATH / 'X_test.csv', index=False)
y_train.to_csv(PROCESSED_PATH / 'y_train.csv', index=False)
y_val.to_csv(PROCESSED_PATH / 'y_val.csv', index=False)
y_test.to_csv(PROCESSED_PATH / 'y_test.csv', index=False)

with open(PROCESSED_PATH / 'features.json', 'w') as f:
    json.dump({'features': FEATURES, 'target': TARGET}, f)

print("6. Обучение модели LightGBM...")
# Лучшие параметры (результат Optuna из экспериментов)
best_params = {
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 100,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': SEED,
}

# Обучаем на train+val для финальной модели
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

model = lgb.LGBMRegressor(**best_params)
model.fit(X_trainval, y_trainval)

print("7. Оценка на тесте...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТЕ:")
print(f"  RMSE: {rmse:,.0f} рупий")
print(f"  MAE:  {mae:,.0f} рупий")
print(f"  R²:   {r2:.4f}")
print(f"{'='*50}\n")

print("8. Сохранение модели...")
joblib.dump(model, MODELS_PATH / 'final_model.pkl')
joblib.dump(FEATURES, MODELS_PATH / 'features.pkl')

print("Модель сохранена в models/final_model.pkl ✅")
print("Список признаков сохранён в models/features.pkl ✅")
