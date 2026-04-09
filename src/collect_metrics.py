# src/collect_metrics.py
import json
import pandas as pd
import os

def main():
    metrics_dir = 'metrics'
    results = []
    
    # Список файлов с метриками
    metric_files = [
        'linear_metrics.json',
        'decision_tree_metrics.json', 
        'catboost_metrics.json',
        'xgboost_metrics.json',
        'mlp_metrics.json'
    ]
    
    for fname in metric_files:
        path = os.path.join(metrics_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'model': data['model'],
                    'mae': data['mae'],
                    'rmse': data['rmse'],
                    'r2': data['r2']
                })
    
    # Создание таблицы
    df = pd.DataFrame(results)
    df = df.sort_values('r2', ascending=False)
    
    # Сохранение
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/summary_metrics.csv', index=False, encoding='utf-8-sig')
    df.to_markdown('reports/summary_metrics.md', index=False)
    
    print("Сводная таблица метрик:")
    print(df.to_string(index=False))
    print("\nСохранено: reports/summary_metrics.csv")
    print("Сохранено: reports/summary_metrics.md")

if __name__ == '__main__':
    main()