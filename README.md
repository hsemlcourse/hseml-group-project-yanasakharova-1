# ✈️ Flight Price Prediction

Регрессионная модель для предсказания стоимости авиабилета.

## Задача
Предсказать цену авиабилета (в рупиях) на основе характеристик рейса.

**Датасет:** [Flight Price Prediction — Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)  
~300 000 строк, 11 исходных колонок.

**Метрика:** RMSE (основная), MAE, R²

## Структура проекта

```
├── data/
│   ├── raw/               # исходный датасет (Clean_Dataset.csv)
│   ├── processed/         # обработанные данные, сплиты
│   └── plots/             # графики
├── notebooks/
│   ├── 01_eda.ipynb           # EDA и визуализации
│   ├── 02_preprocessing.ipynb # очистка и feature engineering
│   └── 03_baseline.ipynb      # baseline модель
├── src/
│   └── utils.py           # вспомогательные функции
├── requirements.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

1. Скачай датасет с Kaggle и положи в `data/raw/Clean_Dataset.csv`
2. Запускай ноутбуки по порядку: 01 → 02 → 03

## Признаки

| Признак | Описание |
|---|---|
| airline | Авиакомпания |
| source_city | Город отправления |
| destination_city | Город прибытия |
| departure_time | Время суток вылета |
| arrival_time | Время суток прилёта |
| stops | Количество пересадок |
| class | Класс обслуживания |
| duration | Длительность полёта (часы) |
| days_left | Дней до вылета |
| **price** | **ТАРГЕТ: цена билета** |
