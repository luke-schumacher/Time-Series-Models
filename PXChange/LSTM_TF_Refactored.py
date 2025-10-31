import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Constants and Configuration ---

MAX_SEQ_LEN = 128
BASE_DATA_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data'
BASE_PREDICTIONS_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange' # Where proportion predictions are stored

# --- 2. Data Loading and Preprocessing ---

def load_and_prepare_ensemble_data(dataset_id):
    """
    Loads predicted proportions and the preprocessed raw data for a given dataset,
    aligning them for ensemble model training.
    """
    proportions_file = os.path.join(BASE_PREDICTIONS_DIR, f'prediction_{dataset_id}_proportions_refactored.csv')
    preprocessed_data_file = os.path.join(BASE_DATA_DIR, dataset_id, f'preprocessed_{dataset_id}.csv')

    if not os.path.exists(proportions_file):
        print(f"❌ Error: Proportions file not found at '{proportions_file}' for dataset {dataset_id}")
        return None, None, None, None, None
    if not os.path.exists(preprocessed_data_file):
        print(f"❌ Error: Preprocessed data file not found at '{preprocessed_data_file}' for dataset {dataset_id}")
        return None, None, None, None, None

    props_df = pd.read_csv(proportions_file)
    processed_raw_df = pd.read_csv(preprocessed_data_file)
    
    # Ensure 'true_total_time' is available from the preprocessed data
    if 'true_total_time' not in processed_raw_df.columns:
        print(f"❌ Error: 'true_total_time' column not found in {preprocessed_data_file}")
        return None, None, None, None, None

    # Extract true total times from the preprocessed raw data
    total_times = processed_raw_df.groupby('SeqOrder')['true_total_time'].first() # 'first' because it's constant per SeqOrder

    # --- Prepare data for the Models ---
    X_sequences, X_num_steps, X_stats = [], [], []
    
    for seq_order, g in props_df.groupby('SeqOrder'):
        proportions = g['predicted_proportion'].values
        X_sequences.append(proportions.reshape(-1, 1))
        X_num_steps.append(len(g))
        
        # --- Feature Engineering: Create a richer set of statistical features ---
        stats = [
            np.mean(proportions), np.std(proportions), np.max(proportions),
            np.percentile(proportions, 25), np.median(proportions), np.percentile(proportions, 75)
        ]
        X_stats.append(stats)
    
    y_total_times_seq_order = props_df['SeqOrder'].unique()
    y_sequences = np.array([total_times.get(seq_id, 0) for seq_id in y_total_times_seq_order])

    X_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
        X_sequences, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32'
    )
    
    X_steps_arr = np.array(X_num_steps, dtype='float32').reshape(-1, 1)
    X_stats_arr = np.array(X_stats, dtype='float32')

    print(f"Successfully prepared ensemble data for dataset {dataset_id} with {len(X_padded_seq)} sequences.")
    
    return X_padded_seq, X_steps_arr, X_stats_arr, y_sequences.reshape(-1, 1), props_df

# --- 3. Model Architectures ---

def build_statistical_model(scalar_shape, stats_shape):
    """Builds a simple MLP model based on high-level statistical features."""
    scalar_input = layers.Input(shape=scalar_shape, name='scalar_input')
    stats_input = layers.Input(shape=stats_shape, name='stats_input')
    
    concatenated = layers.concatenate([scalar_input, stats_input])
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, name='stat_output')(x)
    
    model = tf.keras.Model(inputs=[scalar_input, stats_input], outputs=outputs, name="StatisticalModel")
    return model

def build_sequential_model(sequence_shape):
    """Builds a powerful sequential model that ONLY sees the proportions."""
    sequence_input = layers.Input(shape=sequence_shape, name='sequence_input')
    
    masked_sequence = layers.Masking(mask_value=0.)(sequence_input)
    gru_out = layers.Bidirectional(layers.GRU(128, return_sequences=True))(masked_sequence)
    attention_out = layers.MultiHeadAttention(num_heads=8, key_dim=256)(query=gru_out, value=gru_out, key=gru_out)
    context_vector = layers.GlobalAveragePooling1D()(attention_out)
    
    x = layers.Dense(128, activation='relu')(context_vector)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, name='seq_output')(x)
    
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs, name="SequentialModel")
    return model

def build_meta_model(n_features):
    """Builds the meta-learner model."""
    meta_input = layers.Input(shape=(n_features,), name='meta_input')
    x = layers.Dense(32, activation='relu')(meta_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, name='meta_output')(x)
    
    model = tf.keras.Model(inputs=meta_input, outputs=outputs, name="MetaModel")
    return model

# --- 4. Visualization Function ---

def create_visualizations(results_df, output_dir='visualizations', title_prefix=""):
    """Generates and saves plots comparing true vs. predicted total time."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    true_times, predicted_times = results_df['true_total_time'], results_df['predicted_total_time']
    plt.figure(figsize=(10, 10))
    plt.scatter(true_times, predicted_times, alpha=0.6)
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.xlabel("True Total Time (seconds)"), plt.ylabel("Predicted Total Time (seconds)"), plt.title(f"{title_prefix}True vs. Predicted Total Time"), plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}true_vs_predicted_scatter.png')), plt.close()
    errors = predicted_times - true_times
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Prediction Error (Predicted - True)"), plt.ylabel("Frequency"), plt.title(f"{title_prefix}Distribution of Prediction Errors"), plt.grid(True), plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}prediction_error_histogram.png')), plt.close()
    print(f"✅ Basic visualizations saved to '{output_dir}' directory.")

def create_advanced_visualizations(results_df, output_dir='visualizations', title_prefix=""):
    """Generates and saves advanced diagnostic plots."""
    true_times, predicted_times = results_df['true_total_time'], results_df['predicted_total_time']
    residuals = true_times - predicted_times
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predicted_times, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--'), plt.xlabel("Predicted Total Time (seconds)"), plt.ylabel("Residuals (True - Predicted)"), plt.title(f"{title_prefix}Residuals vs. Predicted Values"), plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}residuals_vs_predicted.png')), plt.close()
    plt.figure(figsize=(10, 6))
    sns.histplot(true_times, color="blue", label='True Values', kde=True, stat="density", linewidth=0)
    sns.histplot(predicted_times, color="red", label='Predicted Values', kde=True, stat="density", linewidth=0)
    plt.title(f"{title_prefix}Distribution of Predicted vs. True Values"), plt.xlabel("Total Time (seconds)"), plt.legend()
    plt.savefig(os.path.join(output_dir, f'{title_prefix.replace(" ", "_").lower()}predicted_vs_true_distribution.png')), plt.close()
    print(f"✅ Advanced visualizations saved to '{output_dir}' directory.")

# --- 5. Training and Prediction Orchestration ---

def main():
    """Main function to run the data processing, training, and prediction for all datasets."""
    
    # Get list of dataset IDs from the data directory
    dataset_ids = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    for dataset_id in dataset_ids:
        print(f"\n--- Processing dataset: {dataset_id} with LSTM Ensemble Model ---")
        
        output_predictions_file = os.path.join(BASE_PREDICTIONS_DIR, f'prediction_{dataset_id}_total_time_refactored.csv')
        
        X_seq, X_steps, X_stats, y, props_df = load_and_prepare_ensemble_data(dataset_id)
        if X_seq is None:
            print(f"Skipping dataset {dataset_id} due to data loading/preparation errors.")
            continue

        X_seq_train, X_seq_val, X_steps_train, X_steps_val, X_stats_train, X_stats_val, y_train, y_val = train_test_split(
            X_seq, X_steps, X_stats, y, test_size=0.2, random_state=42
        )
        
        scaler_steps = StandardScaler(); X_steps_train_scaled, X_steps_val_scaled = scaler_steps.fit_transform(X_steps_train), scaler_steps.transform(X_steps_val)
        scaler_stats = StandardScaler(); X_stats_train_scaled, X_stats_val_scaled = scaler_stats.fit_transform(X_stats_train), scaler_stats.transform(X_stats_val)
        scaler_y = StandardScaler(); y_train_scaled, y_val_scaled = scaler_y.fit_transform(y_train), scaler_y.transform(y_val)
        
        # --- Train Statistical Model ---
        print(f"\n--- Training Statistical Model for {dataset_id} ---")
        stat_model = build_statistical_model((1,), X_stats_train.shape[1:])
        stat_model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        stat_model.fit(
            [X_steps_train_scaled, X_stats_train_scaled], y_train_scaled,
            validation_data=([X_steps_val_scaled, X_stats_val_scaled], y_val_scaled),
            epochs=300, batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)],
            verbose=0
        )
        
        # --- Train Sequential Model ---
        print(f"\n--- Training Sequential Model for {dataset_id} ---")
        seq_model = build_sequential_model(X_seq_train.shape[1:])
        seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
        seq_model.fit(
            X_seq_train, y_train_scaled,
            validation_data=(X_seq_val, y_val_scaled),
            epochs=500, batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75, restore_best_weights=True)],
            verbose=0
        )
        
        # --- Create Training Data for Meta-Model ---
        stat_preds_val = stat_model.predict([X_steps_val_scaled, X_stats_val_scaled])
        seq_preds_val = seq_model.predict(X_seq_val)
        meta_X_train = np.hstack((stat_preds_val, seq_preds_val, X_steps_val_scaled, X_stats_val_scaled))
        meta_y_train = y_val_scaled
        
        # --- Train Meta-Model ---
        print(f"\n--- Training Meta-Model for {dataset_id} ---")
        meta_model = build_meta_model(meta_X_train.shape[1])
        meta_model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        # meta_model.summary() # Optional: print summary for each model
        meta_model.fit(
            meta_X_train, meta_y_train,
            epochs=200, batch_size=16,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)],
            verbose=0 # Set to 1 for more verbose output during meta-model training
        )
        
        # --- Generate Final Ensemble Predictions ---
        print(f"\n--- Generating Final Ensemble Predictions for {dataset_id} ---")
        X_steps_scaled = scaler_steps.transform(X_steps)
        X_stats_scaled = scaler_stats.transform(X_stats)
        stat_preds_scaled = stat_model.predict([X_steps_scaled, X_stats_scaled])
        seq_preds_scaled = seq_model.predict(X_seq)
        
        meta_X_final = np.hstack((stat_preds_scaled, seq_preds_scaled, X_steps_scaled, X_stats_scaled))
        final_preds_scaled = meta_model.predict(meta_X_final)
        
        predicted_times = scaler_y.inverse_transform(final_preds_scaled).flatten()
        
        # --- Final Output ---
        seq_order_to_time = dict(zip(props_df['SeqOrder'].unique(), predicted_times))
        props_df['predicted_total_time'] = np.nan
        
        # Ensure 'Step' column exists for indexing
        if 'Step' not in props_df.columns:
            props_df['Step'] = props_df.groupby('SeqOrder').cumcount()
            
        # Find the index of the last step for each sequence to assign the total time prediction
        # This assumes the total time is associated with the last step of a sequence
        last_step_indices = props_df.groupby('SeqOrder')['Step'].idxmax()
        
        for seq_order, idx in last_step_indices.items():
            if seq_order in seq_order_to_time:
                props_df.loc[idx, 'predicted_total_time'] = seq_order_to_time[seq_order]

        props_df.to_csv(output_predictions_file, index=False)
        print(f"✅ Final predictions for {dataset_id} saved to '{output_predictions_file}'")

        results_df = pd.DataFrame({
            'SeqOrder': props_df['SeqOrder'].unique(),
            'true_total_time': scaler_y.inverse_transform(y).flatten(),
            'predicted_total_time': predicted_times
        })
        
        create_visualizations(results_df, output_dir=os.path.join('visualizations', dataset_id), title_prefix=f"Ensemble {dataset_id} ")
        create_advanced_visualizations(results_df, output_dir=os.path.join('visualizations', dataset_id), title_prefix=f"Ensemble {dataset_id} ")

        print(f"\n--- Sample of Final Predictions for {dataset_id} ---")
        if not props_df.empty: print(props_df[props_df['SeqOrder'] == props_df['SeqOrder'].iloc[0]])

if __name__ == "__main__":
    main()