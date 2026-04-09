import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
import xgboost as xgb
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

    # Обучение XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("XGBoost обучен")

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
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nТоп признаков по важности (XGBoost):")
    print(feature_importance.to_string(index=False))

    # Визуализация Feature Importance
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][::-1], 
             feature_importance['importance'][::-1])
    plt.xlabel('Важность признака')
    plt.ylabel('Признак')
    plt.title('XGBoost: Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранен: reports/xgboost_feature_importance.png")

    # Сохранение модели и метрик
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    joblib.dump(model, 'models/xgboost_model.pkl')
    
    metrics = {
        'model': 'xgboost',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in feature_importance.iterrows()
        }
    }
    with open('metrics/xgboost_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Модель сохранена: models/xgboost_model.pkl")
    print("Метрики сохранены: metrics/xgboost_metrics.json")

if __name__ == '__main__':
    main()