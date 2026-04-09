import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def load_params(params_path='params.yaml'):
    with open(params_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. Загрузка параметров
    params = load_params()
    pp = params['preprocessing']
    target_col = pp['target_col']
    numeric_features = pp['numeric_features']
    categorical_features = pp['categorical_features']
    drop_features = pp['drop_features']
    log_transform = pp.get('log_transform_target', False)

    print(f"Загрузка данных: data/raw/housing.csv")
    df = pd.read_csv('data/raw/housing.csv')
    print(f"Исходный размер: {df.shape}")

    # 2. Отделение целевой переменной и удаление лишних признаков
    y = df[target_col].copy()
    cols_to_drop = [target_col] + [c for c in drop_features if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Логарифмирование целевой переменной
    if log_transform:
        y = np.log1p(y)
        print("Применено: y = log1p(y)")

    # 4. Предобработка признаков
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

    X_prepared = preprocessor.fit_transform(X)

    # Имена признаков после OHE
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_feature_names

    # 5. Сохранение результатов
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    processed_df = pd.DataFrame(X_prepared, columns=feature_names)
    processed_df['target'] = y.values
    processed_df.to_csv('data/processed/prepared_data.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    print(f"Сохранено: data/processed/prepared_data.csv ({processed_df.shape})")
    print(f"Сохранено: models/preprocessor.pkl")
    print(f"Итоговые признаки: {feature_names}")
    print("Этап предобработки успешно завершён!")

if __name__ == '__main__':
    main()