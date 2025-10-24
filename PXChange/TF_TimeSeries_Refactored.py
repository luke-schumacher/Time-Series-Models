import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Constants and Configuration ---

MAX_SEQ_LEN = 128
FEATURE_COLUMNS = [
    'sourceID', 'PTAB', 'BodyGroup_from', 'BodyGroup_to',
    'Position_encoded', 'Direction_encoded'
]
BASE_DATA_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data'
BASE_PREDICTIONS_DIR = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange' # Where proportion predictions are stored

# --- 2. Data Loading and Preprocessing ---

def load_and_prepare_sequences(df):
    """
    Prepares sequences from the preprocessed DataFrame, calculating step durations
    and proportions. This function assumes the input DataFrame `df` is already
    encoded and has 'timediff', 'SeqOrder', and 'true_total_time' columns.
    """
    # Ensure 'Step' column exists. It should be created during preprocessing.
    if 'Step' not in df.columns:
        df['Step'] = df.groupby('SeqOrder').cumcount()
    
    # Ensure 'step_duration' and 'true_proportion' are calculated if not already present
    if 'step_duration' not in df.columns:
        df['step_duration'] = df.groupby('SeqOrder')['timediff'].diff().fillna(df['timediff'])
        df['step_duration'] = df['step_duration'].clip(lower=0)
        end_marker_step = df[df['sourceID'] == 10].groupby('SeqOrder')['Step'].first()
        df['end_marker_step'] = df['SeqOrder'].map(end_marker_step)
        df.loc[df['Step'] > df['end_marker_step'], 'step_duration'] = 0
    
    if 'true_total_time' not in df.columns:
        df['true_total_time'] = df.groupby('SeqOrder')['step_duration'].transform('sum')

    if 'true_proportion' not in df.columns:
        df['true_proportion'] = df['step_duration'] / (df['true_total_time'] + 1e-9)

    # Group data by sequence for the model
    grouped = df.groupby('SeqOrder')
    sequences = []
    proportions = []
    
    print(f"Processing {len(grouped)} sequences...")

    for _, group in grouped:
        seq_features = group[FEATURE_COLUMNS].values
        seq_proportions = group['true_proportion'].values.reshape(-1, 1)
        
        sequences.append(seq_features)
        proportions.append(seq_proportions)

    return sequences, proportions, df

# --- 3. Transformer Model Architecture ---

class PositionalEmbedding(layers.Layer):
    """Adds positional information to the input embeddings."""
    def __init__(self, max_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.pos_encoding.shape[1],
            'embed_dim': self.pos_encoding.shape[2]
        })
        return config

    def positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :] 
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerEncoder(layers.Layer):
    """Transformer Encoder Block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate
        })
        return config

    def call(self, inputs, training=False):
        # The mask is implicitly passed through the layers.
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class MaskedSoftmax(layers.Layer):
    """Applies softmax activation while respecting the mask."""
    def __init__(self, **kwargs):
        super(MaskedSoftmax, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.keras.activations.softmax(inputs, axis=1)

        # Expand mask dimensions to match inputs
        mask = tf.expand_dims(mask, -1)
        
        # Set logits for masked steps to a large negative number
        masked_inputs = tf.where(mask, inputs, -1e9)
        
        return tf.keras.activations.softmax(masked_inputs, axis=1)
        
def build_transformer_model(input_shape, num_heads=4, ff_dim=32, embed_dim=32, num_transformer_blocks=2):
    """
    Builds the single-output Transformer model for proportion prediction.
    """
    num_features = input_shape[-1]
    
    inputs = layers.Input(shape=(None, num_features), name="input_features")
    
    # This layer creates a mask that is passed to all subsequent layers.
    # It masks timesteps where all features are 0 (our padding value).
    masking_layer = layers.Masking(mask_value=0.)(inputs)
    
    dense_proj = layers.Dense(embed_dim, activation="relu")(masking_layer)
    x = PositionalEmbedding(max_len=MAX_SEQ_LEN, embed_dim=embed_dim)(dense_proj)
    
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)
    
    # --- Output Branch: Proportions ---
    time_step_logits = layers.Dense(1, name="time_step_logits")(x)
    proportions_output = MaskedSoftmax(name="proportions_output")(time_step_logits)
    
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=proportions_output
    )
    return model

# --- 4. Visualization Functions ---

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
    """Main function to run the data processing, training, and prediction."""
    
    # Get list of dataset IDs from the data directory
    dataset_ids = [d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]

    for dataset_id in dataset_ids:
        print(f"\n--- Processing dataset: {dataset_id} with Transformer Model ---")
        
        preprocessed_data_file = os.path.join(BASE_DATA_DIR, dataset_id, f'preprocessed_{dataset_id}.csv')
        output_proportions_file = os.path.join(BASE_PREDICTIONS_DIR, f'prediction_{dataset_id}_proportions_refactored.csv')
        
        if not os.path.exists(preprocessed_data_file):
            print(f"❌ Error: Preprocessed data file not found at '{preprocessed_data_file}'. Skipping dataset {dataset_id}.")
            continue

        # --- Step 1: Load preprocessed data ---
        processed_df = pd.read_csv(preprocessed_data_file)
        
        # --- Step 2: Prepare sequences for the Transformer model ---
        # min_timediff and max_timediff are no longer needed here as true_total_time is already calculated
        sequences, proportions, processed_df = load_and_prepare_sequences(processed_df)
        if sequences is None:
            print(f"Skipping dataset {dataset_id} due to sequence preparation errors.")
            continue

        # --- Prepare data for training and prediction ---
        sequence_indices = np.arange(len(sequences))
        train_indices, val_indices = train_test_split(sequence_indices, test_size=0.2, random_state=42)

        X_train_unpadded = [sequences[i] for i in train_indices]
        y_prop_train_unpadded = [proportions[i] for i in train_indices]
        
        X_val_unpadded = [sequences[i] for i in val_indices]
        y_prop_val_unpadded = [proportions[i] for i in val_indices]

        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train_unpadded, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
        y_prop_train = tf.keras.preprocessing.sequence.pad_sequences(y_prop_train_unpadded, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
        X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val_unpadded, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
        y_prop_val = tf.keras.preprocessing.sequence.pad_sequences(y_prop_val_unpadded, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
        
        X_all_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
        
        print(f"\nData shapes (Train): X={X_train.shape}, y_proportions={y_prop_train.shape}")
        print(f"Data shapes (Val):   X={X_val.shape}, y_proportions={y_prop_val.shape}")

        input_shape = X_train.shape[1:]
        model = build_transformer_model(input_shape)
        
        model.compile(
            optimizer="adam", 
            loss=tf.keras.losses.KLDivergence()
        )
        model.summary()
        
        print("\n--- Starting Model Training ---")
        model.fit(
            X_train, 
            y_prop_train,
            validation_data=(X_val, y_prop_val),
            epochs=50,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )
        print("--- Model Training Finished ---\\n")

        # --- Generate Predictions and Create Final Output ---
        print("--- Generating predictions for the entire dataset ---")
        pred_proportions_padded = model.predict(X_all_padded)
        
        # Add the predicted proportions back to the original dataframe for easy output generation
        processed_df['predicted_proportion'] = 0.0
        
        unique_seq_orders = processed_df['SeqOrder'].unique()
        
        for i, seq_order_val in enumerate(unique_seq_orders):
            seq_indices = processed_df[processed_df['SeqOrder'] == seq_order_val].index
            actual_len = len(seq_indices)
            pred_props = pred_proportions_padded[i, :actual_len, 0]
            processed_df.loc[seq_indices, 'predicted_proportion'] = pred_props
                
        # For evaluation within this notebook, let's calculate a 'reconstructed_step_duration' and 'reconstructed_total_time'
        # using the true total time for scaling, to see how well the proportions align.
        processed_df['reconstructed_step_duration'] = processed_df['predicted_proportion'] * processed_df['true_total_time']
        processed_df['reconstructed_total_time'] = processed_df.groupby('SeqOrder')['reconstructed_step_duration'].transform('sum')

        # Select and order columns for the final output file for clarity and verification
        output_columns = [
            'SeqOrder',
            'Step',
            'sourceID',
            'timediff', # Original cumulative timediff for verification
            'step_duration', # The calculated individual step duration
            'true_proportion',
            'predicted_proportion',
            'true_total_time',
            'reconstructed_total_time'
        ]
        
        final_df = processed_df[output_columns]

        final_df.to_csv(output_proportions_file, index=False)
        print(f"✅ Predictions for all sequences of dataset {dataset_id} saved to '{output_proportions_file}'")

        print("\n--- Sample of Predictions ---")
        print(final_df.head(20))

        print("\n--- Verifying Predicted Proportions Sum to 1 (for first 5 sequences) ---")
        print(final_df.groupby('SeqOrder')['predicted_proportion'].sum().head())
        
        print("\n--- Verifying True Proportions Sum to 1 (for first 5 sequences) ---")
        print(final_df.groupby('SeqOrder')['true_proportion'].sum().head())

        # --- Visualizations for reconstructed total time ---
        results_df_for_viz = final_df[['SeqOrder', 'true_total_time', 'reconstructed_total_time']].drop_duplicates(subset=['SeqOrder']).copy()
        results_df_for_viz.rename(columns={'reconstructed_total_time': 'predicted_total_time'}, inplace=True)
        create_visualizations(results_df_for_viz, output_dir=os.path.join('visualizations', dataset_id), title_prefix=f"Transformer Proportions {dataset_id} ")
        create_advanced_visualizations(results_df_for_viz, output_dir=os.path.join('visualizations', dataset_id), title_prefix=f"Transformer Proportions {dataset_id} ")

if __name__ == "__main__":
    main()