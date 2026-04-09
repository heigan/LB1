# src/train_decision_tree.py
import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

    # 1. Загрузка данных
    print("Загрузка: data/processed/prepared_data.csv")
    df = pd.read_csv('data/processed/prepared_data.csv')
    X = df.drop(columns=['target'])
    y = df['target']

    # 2. Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Обучение дерева (ограничим глубину для наглядности)
    model = DecisionTreeRegressor(
        max_depth=4,           # чтобы дерево поместилось на рисунок
        min_samples_split=20,  # защита от переобучения
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Дерево решений обучено")

    # 4. Метрики
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n Метрики на тесте:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R²   : {r2:.4f}")

    # 5. Визуализация первых узлов (сохраняем в файл)
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=X.columns.tolist(),
              filled=True, 
              rounded=True,
              fontsize=10,
              max_depth=3)  # рисуем только первые 3 уровня
    plt.title('Дерево решений: первые узлы (Задача 4)')
    plt.tight_layout()
    plt.savefig('reports/decision_tree_nodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" График сохранён: reports/decision_tree_nodes.png")

    # 6. Сохранение модели и метрик
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    joblib.dump(model, 'models/decision_tree_model.pkl')
    
    metrics = {
        'model': 'decision_tree',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'max_depth': 4,
        'n_nodes': model.tree_.node_count
    }
    with open('metrics/decision_tree_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(" Модель: models/decision_tree_model.pkl")
    print(" Метрики: metrics/decision_tree_metrics.json")

if __name__ == '__main__':
    main()