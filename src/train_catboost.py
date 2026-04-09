import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calc_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }

def main():
    params = load_params()
    random_state = params.get('random_state', 42)

    df = pd.read_csv('data/processed/prepared_data.csv')
    X = df.drop(columns=['target'])
    y = df['target']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )

    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, 
        loss_function='RMSE', random_seed=random_state, verbose=False
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = {
        "model": "catboost",
        "train": calc_metrics(y_train, y_train_pred),
        "val": calc_metrics(y_val, y_val_pred),
        "test": calc_metrics(y_test, y_test_pred),
        "feature_importance": {
            col: float(imp) for col, imp in zip(X.columns, model.get_feature_importance())
        }
    }

    print("CatBoost обучен")
    for split_name, m in [("Train", metrics["train"]), ("Val", metrics["val"]), ("Test", metrics["test"])]:
        print(f"{split_name} -> MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f}")

    os.makedirs('reports', exist_ok=True)
    fi = pd.DataFrame({'feature': X.columns, 'importance': model.get_feature_importance()}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(fi['feature'][::-1], fi['importance'][::-1])
    plt.xlabel('Важность признака')
    plt.ylabel('Признак')
    plt.title('CatBoost: Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/catboost_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    model.save_model('models/catboost_model.cbm')

    with open('metrics/catboost_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()