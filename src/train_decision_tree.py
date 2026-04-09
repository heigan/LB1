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

    model = DecisionTreeRegressor(max_depth=4, min_samples_split=20, random_state=random_state)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = {
        "model": "decision_tree",
        "train": calc_metrics(y_train, y_train_pred),
        "val": calc_metrics(y_val, y_val_pred),
        "test": calc_metrics(y_test, y_test_pred),
        "max_depth": 4,
        "n_nodes": int(model.tree_.node_count)
    }

    print("Дерево решений обучено")
    for split_name, m in [("Train", metrics["train"]), ("Val", metrics["val"]), ("Test", metrics["test"])]:
        print(f"{split_name} -> MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f}")

    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)
    plt.title('Дерево решений: первые узлы')
    plt.tight_layout()
    plt.savefig('reports/decision_tree_nodes.png', dpi=150, bbox_inches='tight')
    plt.close()

    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    joblib.dump(model, 'models/decision_tree_model.pkl')

    with open('metrics/decision_tree_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()