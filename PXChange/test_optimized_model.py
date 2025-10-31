"""Quick test of the optimized model on a single dataset"""
import sys
sys.path.insert(0, 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange')

from LSTM_TF_Optimized import load_and_prepare_enhanced_data, build_enhanced_total_time_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Test on dataset 176401 (has good data)
dataset_id = '176401'

print(f"Testing optimized model on dataset {dataset_id}...")

# Load data
data = load_and_prepare_enhanced_data(dataset_id)
if data is None:
    print("Failed to load data!")
    sys.exit(1)

X_features, X_props, X_meta, y, merged_df = data

# Split
indices = np.arange(len(X_features))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_feat_train, X_feat_val = X_features[train_idx], X_features[val_idx]
X_prop_train, X_prop_val = X_props[train_idx], X_props[val_idx]
X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Scale
meta_scaler = StandardScaler()
X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
X_meta_val_scaled = meta_scaler.transform(X_meta_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)

# Build model
print("Building model...")
model = build_enhanced_total_time_model(
    seq_features_shape=X_feat_train.shape[1:],
    seq_proportions_shape=X_prop_train.shape[1:],
    metadata_shape=X_meta_train_scaled.shape[1:]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='huber',
    metrics=['mae']
)

print(f"Training on {len(train_idx)} sequences...")

# Train with fewer epochs for quick test
history = model.fit(
    [X_feat_train, X_prop_train, X_meta_train_scaled],
    y_train_scaled,
    validation_data=([X_feat_val, X_prop_val, X_meta_val_scaled], y_val_scaled),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ],
    verbose=0
)

print("Training complete!")

# Evaluate
X_meta_all_scaled = meta_scaler.transform(X_meta)
predictions_scaled = model.predict([X_features, X_props, X_meta_all_scaled], verbose=0)
predictions = y_scaler.inverse_transform(predictions_scaled).flatten()

# Calculate metrics
true_times = y.flatten()
errors = predictions - true_times
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
mape = np.mean(np.abs(errors / (true_times + 1e-8))) * 100

print("\n" + "="*50)
print(f"Results for dataset {dataset_id}:")
print(f"MAE:  {mae:.2f} seconds")
print(f"RMSE: {rmse:.2f} seconds")
print(f"MAPE: {mape:.2f}%")
print("="*50)

print("\nSample predictions (first 10):")
for i in range(min(10, len(predictions))):
    print(f"  True: {true_times[i]:.1f}s | Predicted: {predictions[i]:.1f}s | Error: {errors[i]:.1f}s")
