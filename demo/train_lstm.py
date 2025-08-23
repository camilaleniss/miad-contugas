
"""
Training script: trains one LSTM model *per client* to forecast next-hour Volumen
based on the past `lookback` hours of [Volumen, Temperatura, Presion].
Saves: 
  - models/{CLIENTE}.keras   (TensorFlow SavedModel format)
  - models/{CLIENTE}_scaler.npz (NumPy arrays for scalers)
  - predictions/{CLIENTE}_val_predictions.csv (for quick inspection)
Usage:
  python train_lstm.py --data data/synthetic_training.csv --lookback 24 --val_ratio 0.2 --epochs 40
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import json


def build_model(input_steps: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_steps, n_features)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    df = pd.read_csv(args.data)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(["Cliente", "Fecha"])

    features = ["Volumen", "Temperatura", "Presion"]
    meta = {}

    for client, dfg in df.groupby("Cliente"):
        dfg = dfg.sort_values("Fecha").set_index("Fecha")
        # Ensure hourly frequency (fill missing hours if any)
        dfg = dfg.asfreq("H")
        # Interpolate if needed
        dfg[features] = dfg[features].interpolate(limit_direction="both")

        # Train/Val split by time
        n = len(dfg)
        split = int(n * (1 - args.val_ratio))
        train = dfg.iloc[:split].copy()
        val = dfg.iloc[split - args.lookback:].copy()  # include context for lookback

        # Fit scalers on TRAIN only
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_train = train[features].values
        y_train = train[["Volumen"]].values
        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train)

        # Prepare sequences
        Xs_train, ys_train = make_sequences(X_train_scaled, y_train_scaled.ravel(), args.lookback)

        # Validation sequences
        X_val = val[features].values
        y_val = val[["Volumen"]].values
        X_val_scaled = X_scaler.transform(X_val)
        y_val_scaled = y_scaler.transform(y_val)
        Xs_val, ys_val = make_sequences(X_val_scaled, y_val_scaled.ravel(), args.lookback)

        model = build_model(args.lookback, len(features))
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
        history = model.fit(
            Xs_train, ys_train,
            validation_data=(Xs_val, ys_val),
            epochs=args.epochs, batch_size=args.batch_size, verbose=2, callbacks=cb
        )

        # Save model
        model_path = f"models/{client}.keras"
        model.save(model_path)
        # Save scalers (as npz)
        np.savez(f"models/{client}_scaler.npz",
                 X_min=X_scaler.min_, X_scale=X_scaler.scale_, X_data_min=X_scaler.data_min_, X_data_max=X_scaler.data_max_,
                 y_min=y_scaler.min_, y_scale=y_scaler.scale_, y_data_min=y_scaler.data_min_, y_data_max=y_scaler.data_max_
                 )

        # Predictions on validation to inspect
        y_pred_scaled = model.predict(Xs_val, verbose=0)
        # inverse scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        y_true = y_scaler.inverse_transform(ys_val.reshape(-1,1)).ravel()

        # Align timestamps for validation targets
        # Targets start at val.index[lookback:]
        ts = val.index[args.lookback:][:len(y_true)]
        pred_df = pd.DataFrame({
            "Fecha": ts,
            "Cliente": client,
            "y_true": y_true,
            "y_pred": y_pred
        })
        pred_df.to_csv(f"predictions/{client}_val_predictions.csv", index=False)

        meta[client] = {
            "model_path": model_path,
            "features": features,
            "lookback": args.lookback
        }

    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CSV with columns: Fecha, Presion, Temperatura, Volumen, Cliente")
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()
    main(args)
