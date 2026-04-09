
import json
import pandas as pd
import os

def main():
    metrics_dir = 'metrics'
    # Сопоставление файлов метрик с понятными названиями моделей
    metric_files = {
        'linear_metrics.json': 'Линейная регрессия',
        'decision_tree_metrics.json': 'Дерево решений',
        'catboost_metrics.json': 'CatBoost',
        'xgboost_metrics.json': 'XGBoost',
        'mlp_metrics.json': 'Нейронная сеть (MLP)'
    }

    train_data, val_data, test_data = [], [], []

    for fname, display_name in metric_files.items():
        path = os.path.join(metrics_dir, fname)
        if not os.path.exists(path):
            print(f" Файл не найден: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Собираем метрики для каждой выборки, округляем до 4 знаков
        train_data.append({'Модель': display_name, 'MAE': round(data['train']['mae'], 4), 
                           'RMSE': round(data['train']['rmse'], 4), 'R2': round(data['train']['r2'], 4)})
        val_data.append({'Модель': display_name, 'MAE': round(data['val']['mae'], 4), 
                         'RMSE': round(data['val']['rmse'], 4), 'R2': round(data['val']['r2'], 4)})
        test_data.append({'Модель': display_name, 'MAE': round(data['test']['mae'], 4), 
                          'RMSE': round(data['test']['rmse'], 4), 'R2': round(data['test']['r2'], 4)})

    # Создаём DataFrame и сортируем по R² (от лучшего к худшему)
    df_train = pd.DataFrame(train_data).sort_values('R2', ascending=False).reset_index(drop=True)
    df_val   = pd.DataFrame(val_data).sort_values('R2', ascending=False).reset_index(drop=True)
    df_test  = pd.DataFrame(test_data).sort_values('R2', ascending=False).reset_index(drop=True)

    # Сохранение
    os.makedirs('reports', exist_ok=True)
    splits = {'train': df_train, 'val': df_val, 'test': df_test}
    
    for split_name, df in splits.items():
        df.to_csv(f'reports/metrics_{split_name}.csv', index=False, encoding='utf-8-sig')
        df.to_markdown(f'reports/metrics_{split_name}.md', index=False)
        
        print(f"\n{'='*40}")
        print(f" МЕТРИКИ НА ВЫБОРКЕ {split_name.upper()}")
        print(f"{'='*40}")
        print(df.to_string(index=False))
        
    print("\n Таблицы сохранены в reports/metrics_*.csv и reports/metrics_*.md")

if __name__ == '__main__':
    main()