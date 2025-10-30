import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import json
import pandas as pd

# -------------------------------------------------------------------------
# 1. Load and preprocess dataset
# -------------------------------------------------------------------------
fname = r"C:\austi\School\Documents\synthetic_weather_2015_2020.csv"

if not os.path.exists(fname):
    raise FileNotFoundError(f"Dataset not found at {fname}")

data = pd.read_csv(fname)
print("Loaded synthetic dataset:", data.shape)
print("Columns:", data.columns.tolist()[:10], "...")

lines = data.values
header = data.columns
num_records = len(lines)
num_features = len(header) - 1  

raw_data = lines[:, 1:].astype(float)
temperature = raw_data[:, 1]  

# -------------------------------------------------------------------------
# 2. Plot a sample of the temperature data
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(range(1440), temperature[:1440])
plt.title("First 10 days of synthetic temperature data")
plt.xlabel("Timesteps (10-min intervals)")
plt.ylabel("Temperature (°C)")
plt.show()

# -------------------------------------------------------------------------
# 3. Train/val/test split and normalization
# -------------------------------------------------------------------------
num_train = int(0.5 * num_records)
num_val = int(0.25 * num_records)
num_test = num_records - num_train - num_val

print("num_train:", num_train, "num_val:", num_val, "num_test:", num_test)

mean = raw_data[:num_train].mean(axis=0)
std = raw_data[:num_train].std(axis=0)
raw_data -= mean
raw_data /= std

# -------------------------------------------------------------------------
# 4. Prepare time-series datasets
# -------------------------------------------------------------------------
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 128

train_ds = keras.utils.timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train
)

val_ds = keras.utils.timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=num_train,
    end_index=num_train + num_val
)

test_ds = keras.utils.timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=num_train + num_val,
    end_index=None
)

for x_batch, y_batch in train_ds.take(1):
    print("Batch x shape:", x_batch.shape, "Batch y shape:", y_batch.shape)
    break

# -------------------------------------------------------------------------
# 5. Model definitions
# -------------------------------------------------------------------------
def build_gru_model(input_shape, units=(32, 32)):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i, u in enumerate(units):
        x = layers.GRU(u, return_sequences=(i < len(units) - 1))(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.RMSprop(), loss="mae", metrics=["mae"])
    return model

def build_lstm_model(input_shape, units=(32, 32)):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=(i < len(units) - 1))(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.RMSprop(), loss="mae", metrics=["mae"])
    return model

def build_conv_lstm_model(input_shape, conv_filters=32, kernel_size=5, lstm_units=32):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(conv_filters, kernel_size, padding="causal", activation="relu")(inp)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.LSTM(lstm_units)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.RMSprop(), loss="mae", metrics=["mae"])
    return model

# -------------------------------------------------------------------------
# 6. Train models
# -------------------------------------------------------------------------
sample_shape = (sequence_length, raw_data.shape[-1])
models = {
    "GRU_32x32": build_gru_model(sample_shape, units=(32, 16)),
    "LSTM_32x32": build_lstm_model(sample_shape, units=(32, 16)),
    "Conv1D_LSTM": build_conv_lstm_model(sample_shape),
}

histories = {}
best_val_mae = {}
checkpoint_dir = "./checkpoints_assignment3"
os.makedirs(checkpoint_dir, exist_ok=True)

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    ckpt_path = os.path.join(checkpoint_dir, f"{name}.weights.h5")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_mae", save_best_only=True, save_weights_only=True)
    ]
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    histories[name] = history
    model.load_weights(ckpt_path)
    best_val_mae[name] = min(history.history["val_mae"])
    print(f"Best validation MAE for {name}: {best_val_mae[name]:.4f}")

# -------------------------------------------------------------------------
# 7. Compare models
# -------------------------------------------------------------------------
print("\nValidation MAE summary:")
for name, mae in sorted(best_val_mae.items(), key=lambda x: x[1]):
    print(f"{name}: {mae:.4f}")

best_model_name = min(best_val_mae, key=best_val_mae.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

test_loss, test_mae = best_model.evaluate(test_ds, verbose=2)
print(f"Test MAE: {test_mae:.4f}")

# -------------------------------------------------------------------------
# 8. Plot MAE trends
# -------------------------------------------------------------------------
plt.figure(figsize=(10,6))
for name, h in histories.items():
    plt.plot(h.history["val_mae"], label=f"{name} val_mae")
plt.title("Validation MAE Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------
# 9. Sample predictions
# -------------------------------------------------------------------------
x_samples, y_true = [], []
for xb, yb in test_ds.take(1):
    x_samples = xb.numpy()
    y_true = yb.numpy()
y_pred = best_model.predict(x_samples)

plt.figure(figsize=(10,5))
plt.plot(y_true[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title(f"Sample Predictions - {best_model_name}")
plt.legend()
plt.show()

# -------------------------------------------------------------------------
# 10. Save summary
# -------------------------------------------------------------------------
summary = {
    "best_model": best_model_name,
    "validation_mae": {k: float(v) for k, v in best_val_mae.items()},
    "test_mae": float(test_mae)
}
with open("assignment3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Summary saved to assignment3_summary.json")

