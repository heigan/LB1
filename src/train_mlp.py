# src/train_mlp.py
import pandas as pd
import numpy as np
import yaml
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
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
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    df = pd.read_csv('data/processed/prepared_data.csv')
    X = df.drop(columns=['target']).values
    y = df['target'].values.reshape(-1, 1)

    # Разделение 60% train / 20% val / 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=random_state
    )

    # Масштабирование целевой переменной (только на train, чтобы не было утечки данных)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).flatten()
    y_val_scaled = scaler_y.transform(y_val).flatten()
    y_test_scaled = scaler_y.transform(y_test).flatten()

    # Архитектура MLP
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])

    # Callbacks
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    tb_callback = TensorBoard(
        log_dir='logs/mlp_run',
        histogram_freq=1,          # Сохраняет гистограммы весов и активаций
        write_graph=True,          # Сохраняет вычислительный граф модели
        update_freq='epoch',       # Логирует метрики каждую эпоху
        profile_batch=0            # Отключает профилирование для ускорения
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    print("Обучение нейронной сети...")
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=200,
        batch_size=32,
        callbacks=[tb_callback, early_stop],
        verbose=0
    )
    print("Обучение завершено")

    # Прогнозы и обратное преобразование в исходные единицы
    y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_train_orig = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    metrics = {
        "model": "mlp_neural_network",
        "train": calc_metrics(y_train_orig, y_train_pred),
        "val": calc_metrics(y_val_orig, y_val_pred),
        "test": calc_metrics(y_test_orig, y_test_pred),
        "architecture": {"layers": [64, 32, 16, 1], "activation": "relu", "optimizer": "adam"},
        "epochs_trained": len(history.history['loss'])
    }

    print("Метрики:")
    for split_name, m in [("Train", metrics["train"]), ("Val", metrics["val"]), ("Test", metrics["test"])]:
        print(f"{split_name} -> MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | R2: {m['r2']:.4f}")

    # Кривые обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
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

    # Гистограммы весов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    layer_weights = [layer.get_weights()[0] for layer in model.layers if layer.get_weights()]
    for i, weights in enumerate(layer_weights[:4]):
        axes[i].hist(weights.flatten(), bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Слой {i+1}: mean={weights.mean():.3f}, std={weights.std():.3f}')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/mlp_weight_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Сохранение артефактов
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    model.save('models/mlp_model.keras')

    with open('metrics/mlp_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Логи TensorBoard сохранены: logs/mlp_run/")
    print("Для запуска панели выполните в терминале: tensorboard --logdir=logs/mlp_run")

if __name__ == '__main__':
    main()