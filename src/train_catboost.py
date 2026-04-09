import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    random_state = params.get('random_state', 42)
    test_size = params.get('test_size', 0.2)

    # Загрузка данных
    print("Загрузка: data/processed/prepared_data.csv")
    df = pd.read_csv('data/processed/prepared_data.csv')
    X = df.drop(columns=['target'])
    y = df['target']

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Определение категориальных признаков (для CatBoost)
    # После One-Hot Encoding у нас нет категориальных столбцов,
    # но если бы были, указывали бы их индексы здесь
    cat_features = []

    # Обучение CatBoost
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=random_state,
        verbose=False,
        cat_features=cat_features
    )
    model.fit(X_train, y_train)
    print("CatBoost обучен")

    # Прогнозы и метрики
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nМетрики на тесте:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R2   : {r2:.4f}")

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    print(f"\nТоп-10 признаков по важности (CatBoost):")
    print(feature_importance.head(10).to_string(index=False))

    # Визуализация Feature Importance
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'].head(10)[::-1], 
             feature_importance['importance'].head(10)[::-1])
    plt.xlabel('Важность признака')
    plt.ylabel('Признак')
    plt.title('CatBoost: Feature Importance (Топ-10)')
    plt.tight_layout()
    plt.savefig('reports/catboost_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранен: reports/catboost_feature_importance.png")

    # Сохранение модели и метрик
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    model.save_model('models/catboost_model.cbm')
    
    metrics = {
        'model': 'catboost',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in feature_importance.iterrows()
        }
    }
    with open('metrics/catboost_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Модель сохранена: models/catboost_model.cbm")
    print("Метрики сохранены: metrics/catboost_metrics.json")

if __name__ == '__main__':
    main()