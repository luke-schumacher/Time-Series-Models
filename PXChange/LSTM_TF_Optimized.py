import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---

MAX_SEQ_LEN = 128
FEATURE_COLUMNS = [
    'sourceID', 'PTAB', 'BodyGroup_from', 'BodyGroup_to',
    'Position_encoded', 'Direction_encoded'
]
BASE_DATA_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data'
BASE_PREDICTIONS_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange'

# --- 2. Enhanced Data Loading ---

def load_and_prepare_enhanced_data(dataset_id):
    """
    Loads BOTH predicted proportions AND original sequence features for richer prediction.
    """
    proportions_file = os.path.join(BASE_PREDICTIONS_DIR, f'prediction_{dataset_id}_proportions_refactored.csv')
    preprocessed_data_file = os.path.join(BASE_DATA_DIR, dataset_id, f'preprocessed_{dataset_id}.csv')

    if not os.path.exists(proportions_file):
        print(f">> Error: Proportions file not found at '{proportions_file}'")
        return None
    if not os.path.exists(preprocessed_data_file):
        print(f">> Error: Preprocessed data file not found at '{preprocessed_data_file}'")
        return None

    props_df = pd.read_csv(proportions_file)
    preprocessed_df = pd.read_csv(preprocessed_data_file)

    if 'true_total_time' not in preprocessed_df.columns:
        print(f">> Error: 'true_total_time' not found in preprocessed data")
        return None

    # Get total times from preprocessed data (this is the source of truth)
    total_times_map = preprocessed_df.groupby('SeqOrder')['true_total_time'].first().to_dict()

    # Merge to get all features aligned
    merged_df = pd.merge(
        props_df[['SeqOrder', 'Step', 'predicted_proportion']],
        preprocessed_df,
        on=['SeqOrder'],
        how='left'
    )

    # Prepare sequences
    X_seq_features = []  # Original sequence features
    X_seq_proportions = []  # Predicted proportions
    X_metadata = []  # Sequence-level metadata
    y_total_times = []

    for seq_order, group in merged_df.groupby('SeqOrder'):
        # Get sequence of original features
        seq_features = group[FEATURE_COLUMNS].values
        X_seq_features.append(seq_features)

        # Get predicted proportions
        proportions = group['predicted_proportion'].values.reshape(-1, 1)
        X_seq_proportions.append(proportions)

        # Get sequence-level metadata (constant per sequence)
        first_row = group.iloc[0]
        metadata = [
            first_row.get('Age', 0),
            first_row.get('Weight', 0),
            first_row.get('Height', 0),
            first_row.get('BodyGroup_from', 0),
            first_row.get('BodyGroup_to', 0),
            len(group)  # Sequence length
        ]
        X_metadata.append(metadata)

        # Target - get from the total_times_map
        y_total_times.append(total_times_map.get(seq_order, 0))

    # Pad sequences
    X_seq_features_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_seq_features, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32'
    )
    X_seq_proportions_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_seq_proportions, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32'
    )

    X_metadata_arr = np.array(X_metadata, dtype='float32')
    y = np.array(y_total_times, dtype='float32').reshape(-1, 1)

    print(f">> Prepared enhanced data for {dataset_id}: {len(X_seq_features_padded)} sequences")
    print(f"   Features shape: {X_seq_features_padded.shape}, Proportions: {X_seq_proportions_padded.shape}, Metadata: {X_metadata_arr.shape}")

    return X_seq_features_padded, X_seq_proportions_padded, X_metadata_arr, y, merged_df

# --- 3. Enhanced Model Architecture ---

def build_enhanced_total_time_model(seq_features_shape, seq_proportions_shape, metadata_shape):
    """
    Enhanced model that combines:
    1. Original sequence features (sourceID, PTAB, BodyGroups, etc.)
    2. Predicted proportions from Transformer
    3. Patient/scan metadata (Age, Weight, Height, etc.)
    """

    # Input branches
    seq_features_input = layers.Input(shape=seq_features_shape, name='seq_features')
    seq_proportions_input = layers.Input(shape=seq_proportions_shape, name='seq_proportions')
    metadata_input = layers.Input(shape=metadata_shape, name='metadata')

    # --- Branch 1: Process original sequence features ---
    masked_features = layers.Masking(mask_value=0.)(seq_features_input)

    # Bidirectional LSTM to capture temporal patterns
    lstm1 = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(masked_features)

    # Self-attention to focus on important steps
    attention1 = layers.MultiHeadAttention(
        num_heads=8, key_dim=64, dropout=0.1
    )(lstm1, lstm1)
    attention1 = layers.LayerNormalization()(attention1 + lstm1)

    # Global pooling
    features_context = layers.GlobalAveragePooling1D()(attention1)
    features_max = layers.GlobalMaxPooling1D()(attention1)
    features_combined = layers.concatenate([features_context, features_max])

    # --- Branch 2: Process predicted proportions ---
    masked_proportions = layers.Masking(mask_value=0.)(seq_proportions_input)

    # GRU for proportion patterns
    gru = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, dropout=0.2)
    )(masked_proportions)

    # Attention on proportions
    attention2 = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.1
    )(gru, gru)
    attention2 = layers.LayerNormalization()(attention2 + gru)

    # Global pooling
    prop_context = layers.GlobalAveragePooling1D()(attention2)
    prop_max = layers.GlobalMaxPooling1D()(attention2)
    prop_combined = layers.concatenate([prop_context, prop_max])

    # --- Branch 3: Process metadata ---
    metadata_dense = layers.Dense(32, activation='relu')(metadata_input)
    metadata_dense = layers.Dropout(0.2)(metadata_dense)
    metadata_dense = layers.Dense(16, activation='relu')(metadata_dense)

    # --- Combine all branches ---
    combined = layers.concatenate([features_combined, prop_combined, metadata_dense])

    # Final prediction layers
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, name='total_time_output')(x)

    model = Model(
        inputs=[seq_features_input, seq_proportions_input, metadata_input],
        outputs=output,
        name='EnhancedTotalTimePredictor'
    )

    return model

# --- 4. Visualization Functions ---

def create_visualizations(results_df, output_dir='visualizations', title_prefix=""):
    """Generates comparison plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    true_times = results_df['true_total_time'].values
    predicted_times = results_df['predicted_total_time'].values

    # Scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(true_times, predicted_times, alpha=0.6, s=50)
    lims = [
        min(true_times.min(), predicted_times.min()),
        max(true_times.max(), predicted_times.max())
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect Prediction')
    plt.xlabel("True Total Time (seconds)", fontsize=12)
    plt.ylabel("Predicted Total Time (seconds)", fontsize=12)
    plt.title(f"{title_prefix}True vs Predicted Total Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}scatter.png'), dpi=150)
    plt.close()

    # Error histogram
    errors = predicted_times - true_times
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel("Prediction Error (seconds)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"{title_prefix}Prediction Error Distribution", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}error_histogram.png'), dpi=150)
    plt.close()

    # Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = true_times - predicted_times
    plt.scatter(predicted_times, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Total Time (seconds)", fontsize=12)
    plt.ylabel("Residuals (True - Predicted)", fontsize=12)
    plt.title(f"{title_prefix}Residual Plot", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}residuals.png'), dpi=150)
    plt.close()

    # Calculate metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (true_times + 1e-8))) * 100

    print(f"   >> MAE: {mae:.2f}s | RMSE: {rmse:.2f}s | MAPE: {mape:.2f}%")
    print(f"   >> Visualizations saved to '{output_dir}'")

# --- 5. Training Pipeline ---

def train_and_evaluate(dataset_id):
    """Train enhanced model on a single dataset."""
    print(f"\n{'='*70}")
    print(f">> Processing Dataset: {dataset_id}")
    print(f"{'='*70}")

    output_file = os.path.join(BASE_PREDICTIONS_DIR, f'prediction_{dataset_id}_total_time_refactored.csv')

    # Load data
    data = load_and_prepare_enhanced_data(dataset_id)
    if data is None:
        print(f">> Skipping {dataset_id} due to data loading errors")
        return

    X_features, X_props, X_meta, y, merged_df = data

    # Split data
    indices = np.arange(len(X_features))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_feat_train, X_feat_val = X_features[train_idx], X_features[val_idx]
    X_prop_train, X_prop_val = X_props[train_idx], X_props[val_idx]
    X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale metadata and targets
    meta_scaler = StandardScaler()
    X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
    X_meta_val_scaled = meta_scaler.transform(X_meta_val)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    # Build model
    print("\n>> Building Enhanced Model...")
    model = build_enhanced_total_time_model(
        seq_features_shape=X_feat_train.shape[1:],
        seq_proportions_shape=X_prop_train.shape[1:],
        metadata_shape=X_meta_train_scaled.shape[1:]
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )

    print(f"\n>> Training model on {len(train_idx)} sequences...")

    # Train
    history = model.fit(
        [X_feat_train, X_prop_train, X_meta_train_scaled],
        y_train_scaled,
        validation_data=([X_feat_val, X_prop_val, X_meta_val_scaled], y_val_scaled),
        epochs=200,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=0
    )

    # Predict on all data
    print("\n>> Generating predictions...")
    X_meta_all_scaled = meta_scaler.transform(X_meta)
    predictions_scaled = model.predict([X_features, X_props, X_meta_all_scaled], verbose=0)
    predictions = y_scaler.inverse_transform(predictions_scaled).flatten()

    # Map predictions back to dataframe
    seq_orders = merged_df['SeqOrder'].unique()
    seq_to_pred = dict(zip(seq_orders, predictions))

    merged_df['predicted_total_time'] = merged_df['SeqOrder'].map(seq_to_pred)

    # Save
    output_cols = ['SeqOrder', 'Step', 'sourceID', 'timediff', 'predicted_total_time', 'true_total_time',
                   'BodyGroup_from', 'BodyGroup_to', 'Age', 'Weight', 'Height',
                   'Position_encoded', 'Direction_encoded']
    output_cols = [col for col in output_cols if col in merged_df.columns]
    merged_df[output_cols].to_csv(output_file, index=False)

    print(f">> Predictions saved to: {output_file}")

    # Evaluate
    results_df = pd.DataFrame({
        'SeqOrder': seq_orders,
        'true_total_time': y.flatten(),
        'predicted_total_time': predictions
    })

    viz_dir = os.path.join('visualizations', dataset_id)
    create_visualizations(results_df, output_dir=viz_dir, title_prefix=f"Enhanced_{dataset_id}_")

    return model, history, results_df

# --- 6. Main Execution ---

def main():
    """Train enhanced model on all datasets."""
    dataset_ids = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]

    print(f"\n>> Starting Enhanced Total Time Prediction")
    print(f">> Found {len(dataset_ids)} datasets: {dataset_ids}\n")

    all_results = []

    for dataset_id in dataset_ids:
        try:
            model, history, results = train_and_evaluate(dataset_id)
            if results is not None:
                all_results.append(results)
        except Exception as e:
            print(f"\n>> Error processing {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Overall summary
    if all_results:
        print(f"\n{'='*70}")
        print(f">> OVERALL SUMMARY")
        print(f"{'='*70}")

        combined_results = pd.concat(all_results, ignore_index=True)
        true_all = combined_results['true_total_time'].values
        pred_all = combined_results['predicted_total_time'].values

        mae = np.mean(np.abs(pred_all - true_all))
        rmse = np.sqrt(np.mean((pred_all - true_all)**2))
        mape = np.mean(np.abs((pred_all - true_all) / (true_all + 1e-8))) * 100

        print(f"Overall MAE:  {mae:.2f} seconds")
        print(f"Overall RMSE: {rmse:.2f} seconds")
        print(f"Overall MAPE: {mape:.2f}%")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
