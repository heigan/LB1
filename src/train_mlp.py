import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_params(path='params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    random_state = params.get('random_state', 42)
    test_size = params.get('test_size', 0.2)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    print("Загрузка: data/processed/prepared_data.csv")
    df = pd.read_csv('data/processed/prepared_data.csv')
    X = df.drop(columns=['target']).values
    y = df['target'].values.reshape(-1, 1)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=test_size, random_state=random_state
    )

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    os.makedirs('reports', exist_ok=True)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    print("Обучение нейронной сети...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    print("Обучение завершено")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    print(f"\nМетрики на тесте:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R2   : {r2:.4f}")

    # Кривые обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (MSE)')
    plt.title('Кривые обучения: Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Эпоха')
    plt.ylabel('MAE')
    plt.title('Кривые обучения: MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/mlp_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранен: reports/mlp_learning_curves.png")

    # Гистограммы весов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    layer_weights = [layer.get_weights()[0] for layer in model.layers if layer.get_weights()]
    
    for i, weights in enumerate(layer_weights[:4]):
        axes[i].hist(weights.flatten(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Слой {i+1}: веса (mean={weights.mean():.4f}, std={weights.std():.4f})')
        axes[i].set_xlabel('Значение веса')
        axes[i].set_ylabel('Частота')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/mlp_weight_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранен: reports/mlp_weight_histograms.png")
    
    print("\nИнтерпретация гистограмм весов:")
    for i, weights in enumerate(layer_weights[:4]):
        mean_w = weights.mean()
        std_w = weights.std()
        if std_w < 0.1:
            interpretation = "веса сконцентрированы около нуля — слой может быть недообучен"
        elif std_w > 1.0:
            interpretation = "большой разброс весов — возможна нестабильность градиентов"
        else:
            interpretation = "нормальное распределение весов — обучение стабильно"
        print(f"  Слой {i+1}: mean={mean_w:.4f}, std={std_w:.4f} — {interpretation}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    model.save('models/mlp_model.keras')
    
    metrics = {
        'model': 'mlp_neural_network',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'architecture': {
            'layers': [64, 32, 16, 1],
            'activation': 'relu',
            'optimizer': 'adam',
            'epochs_trained': len(history.history['loss'])
        }
    }
    with open('metrics/mlp_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Модель сохранена: models/mlp_model.keras")
    print("Метрики сохранены: metrics/mlp_metrics.json")

if __name__ == '__main__':
    main()