# src/train_linear.py
import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    random_state = params.get('random_state', 42)
    test_size = params.get('test_size', 0.2)

    # 1. Загрузка подготовленных данных
    print("Загрузка данных: data/processed/prepared_data.csv")
    df = pd.read_csv('data/processed/prepared_data.csv')
    
    # Разделяем признаки и целевую переменную
    target_col = 'target'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. Обучение линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Линейная регрессия обучена")

    # 4. Прогнозы и метрики
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n Метрики на тесте:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R²   : {r2:.4f}")

    # 5. Веса модели
    weights = dict(zip(X.columns, model.coef_))
    print(f"\n Веса признаков: {weights}")

    # 6. Сохранение результатов
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    joblib.dump(model, 'models/linear_model.pkl')
    print("Модель сохранена: models/linear_model.pkl")

    metrics = {
        'model': 'linear_regression',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'weights': {k: float(v) for k, v in weights.items()}
    }
    with open('metrics/linear_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Метрики сохранены: metrics/linear_metrics.json")

if __name__ == '__main__':
    main()