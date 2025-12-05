"""
Module: lstm_builder.py
Mục đích: đóng gói việc tạo, compile và huấn luyện mô hình LSTM vào các hàm tái sử dụng.

Hàm chính:
- build_lstm_model(...) -> trả về một model đã được compile
- get_default_callbacks(...) -> trả về danh sách callbacks giống như trong ví dụ của bạn
- train_model(...) -> chạy model.fit và (tùy chọn) evaluate trên tập test

Sử dụng: mở file này trong canvas, import các hàm rồi gọi.
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LSTM, Dense, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import RobustScaler
import pandas as pd

def split_data(raw_data, train_ratio=0.7, val_ratio=0.15):
    """
    Chia dữ liệu theo thứ tự thời gian để tránh leak.
    train : val : test = 70% : 15% : 15% (mặc định)
    """
    n = len(raw_data)
    train_split = int(n * train_ratio)
    val_split = int(n * (train_ratio + val_ratio))

    train_raw = raw_data[:train_split]
    val_raw = raw_data[train_split:val_split]
    test_raw = raw_data[val_split:]

    print(f"Dữ liệu gốc: {n} dòng")
    print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

    return train_raw, val_raw, test_raw


def scale_data(train_raw, val_raw, test_raw):
    """
    Scale dữ liệu bằng RobustScaler.
    Quan trọng: Chỉ fit trên train để tránh data leakage.
    """
    scaler = RobustScaler()

    # Fit trên Train
    scaler.fit(train_raw)

    # Transform trên cả 3 sets
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    return train_scaled, val_scaled, test_scaled, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(
    n_timesteps: int,
    n_features: int,
    n_outputs: int,
    lstm_units: List[int] = [256, 256],
    dense_units: int = 256,
    leaky_alpha: float = 0.1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Xây dựng và compile một model LSTM giống cấu trúc bạn đưa.

    Trả về: model đã compile (Huber loss + Adam + metrics rmse, mae)
    """

    model = Sequential([
        # Input normalization (đặt input_shape ở layer đầu)
        BatchNormalization(input_shape=(n_timesteps, n_features), name='Batch_Norm_Input'),

        # LSTM layer 1 (return sequences)
        LSTM(lstm_units[0], return_sequences=True, name='LSTM_1'),
        LeakyReLU(alpha=leaky_alpha),
        BatchNormalization(name='Batch_Norm_1'),

        # LSTM layer 2 (không return sequences để nối vào Dense)
        LSTM(lstm_units[1], return_sequences=False, name='LSTM_2'),
        LeakyReLU(alpha=leaky_alpha),
        BatchNormalization(momentum=0.8, name='Batch_Norm_2'),

        # Dense và output
        Dense(dense_units, name='Dense_1'),
        LeakyReLU(alpha=leaky_alpha),
        Dense(n_outputs, activation='linear', name='Returns')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer,
        metrics=[tf.metrics.RootMeanSquaredError(name='rmse'), 'mae']
    )

    return model


def get_default_callbacks(
    filepath: str = 'best_model_lstm.keras',
    lr_factor: float = 0.5,
    lr_patience: int = 10,
    min_lr: float = 1e-6,
    es_patience: int = 20,
    monitor: str = 'val_loss',
) -> List[tf.keras.callbacks.Callback]:
    """Tạo danh sách callbacks giống ví dụ của bạn (checkpoint, lr reducer, early stopping).

    Trả về: [ModelCheckpoint, ReduceLROnPlateau, EarlyStopping]
    """

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=False,
        monitor=monitor,
        mode='min',
        save_best_only=True,
        verbose=1
    )

    lr_reducer = ReduceLROnPlateau(
        monitor=monitor,
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=es_patience,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, lr_reducer, early_stopping]


def train_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,

    epochs: int = 100,
    batch_size: int = 32,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    verbose: int = 1,
) -> Tuple[tf.keras.callbacks.History]:
    """Chạy fit và KHÔNG tự evaluate trên tập test.

    Trả về: (history, None). Đánh giá test tách riêng bằng evaluate_model().
    """

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )


    return history

def build_results_dataframe(scaler, y_pred_scaled, y_test_scaled, df_index, test_start_idx):
    """
    scaler: đã fit trên train_raw (ví dụ RobustScaler)
    y_pred_scaled: numpy array của dự báo (scaled)
    y_test_scaled: numpy array của ground truth (scaled)
    df_index: index gốc chứa cột thời gian — thường là df_pairs_AAPL_AMZN.index
    test_start_idx: vị trí bắt đầu của test set trong df_index (sau train+val + bất kỳ offset nào như SEQ_LENGTH)
    
    Trả về:
      DataFrame với index là ngày (time index), các cột:
       - Actual_Spread: giá trị thực (unscaled)
       - Predicted_Spread: giá trị dự báo (unscaled)
       - Prev_Actual: Actual của ngày trước
       - LSTM_Trend: 1 nếu Predicted_Spread > Prev_Actual, else -1
    """
    # Inverse transform để đưa về giá trị thực
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test_scaled)
    
    # Tạo index thời gian tương ứng
    result_index = df_index[test_start_idx : test_start_idx + len(y_pred)]
    
    df_results = pd.DataFrame({
        'Actual_Spread': y_actual.flatten(),
        'Predicted_Spread': y_pred.flatten()
    }, index=result_index)
    
    # Tính trend
    df_results['Prev_Actual'] = df_results['Actual_Spread'].shift(1)
    df_results['LSTM_Trend'] = np.where(
        df_results['Predicted_Spread'] > df_results['Prev_Actual'],
        1, -1
    )
    
    df_results = df_results.dropna()
    
    return df_results



# Ví dụ sử dụng (chạy trực tiếp file này)


if __name__ == '__main__':
    import numpy as np

    # Dummy shapes -- bạn thay bằng dữ liệu thật
    n_timesteps = 10
    n_features = 5
    n_outputs = 1

    X_train = np.random.randn(200, n_timesteps, n_features)
    y_train = np.random.randn(200, n_outputs)
    X_val = np.random.randn(50, n_timesteps, n_features)
    y_val = np.random.randn(50, n_outputs)
    X_test = np.random.randn(50, n_timesteps, n_features)
    y_test = np.random.randn(50, n_outputs)

    model = build_lstm_model(n_timesteps, n_features, n_outputs)
    callbacks = get_default_callbacks()

    # 1) Huấn luyện (không evaluate tự động)
    history, _ = train_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        epochs=5, batch_size=32, callbacks=callbacks
    )


    
