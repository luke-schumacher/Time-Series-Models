import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# Constants and Encoding
START_TOKEN = 13
END_TOKEN = 14
ENCODING_LEGEND = {
    'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7, 'MRI_FRR_34': 8, 'MRI_MPT_1005': 9,
    'MRI_MSR_100': 10, 'MRI_MSR_104': 11, 'MRI_MSR_21': 12,
    'START': START_TOKEN, 'END': END_TOKEN
}
reverse_encoding = {v: k for k, v in ENCODING_LEGEND.items()}

# Define valid source IDs for filtering (excluding START and END tokens)
VALID_SOURCE_IDS = set([k for k in ENCODING_LEGEND.keys() if k not in ['START', 'END']])

# %%
def load_and_preprocess_data(data_file):
    """
    Loads and preprocesses data from a CSV file, filtering out invalid sourceIDs.
    Splits data into sequences based on SeqOrder.
    """
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)

    all_sequences_tokens = []
    all_sequences_times = []
    all_sequences_sourceids = []

    current_tokens = []
    current_times = []
    current_sourceids = []

    for idx, row in data.iterrows():
        seq_order = row['SeqOrder']
        s_id = str(row['sourceID']) # Ensure s_id is string for lookup
        t_diff = float(row['timediff'])

        # Filter: Only process rows with valid sourceIDs
        if s_id not in VALID_SOURCE_IDS:
            # print(f"Skipping row with invalid sourceID: {s_id}") # Optional: uncomment for debugging
            continue

        if seq_order == 0 and current_tokens:
            # Finalize previous sequence if it exists and we are starting a new one
            token_seq = [START_TOKEN] + [int(ENCODING_LEGEND[x]) for x in current_tokens] + [END_TOKEN]
            time_seq = [0.0] + current_times

            all_sequences_tokens.append(token_seq)
            all_sequences_times.append(time_seq)
            all_sequences_sourceids.append(current_sourceids)

            # Reset for the new sequence
            current_tokens = []
            current_times = []
            current_sourceids = []

        # Append current valid token and time difference
        current_tokens.append(s_id)
        current_times.append(t_diff)
        current_sourceids.append(s_id) # Store the original string sourceID

    # Add the last sequence if data is not empty
    if current_tokens:
        token_seq = [START_TOKEN] + [int(ENCODING_LEGEND[x]) for x in current_tokens] + [END_TOKEN]
        time_seq = [0.0] + current_times

        all_sequences_tokens.append(token_seq)
        all_sequences_times.append(time_seq)
        all_sequences_sourceids.append(current_sourceids)

    print(f"Loaded {len(all_sequences_tokens)} sequences.")
    return all_sequences_tokens, all_sequences_times, all_sequences_sourceids

# %%
def prepare_training_data(sequences_tokens, sequences_times):
    """
    Prepares sequences for transformer training, including padding and masks.
    Calculates target cumulative times and total times.
    """
    X_list, Y_list, masks_list, total_times_list = [], [], [], []

    for tokens, times in zip(sequences_tokens, sequences_times):
        # Ensure sequence has at least START and END tokens plus one event
        if len(tokens) < 3:
            # print(f"Skipping short sequence with {len(tokens)} tokens.") # Optional: uncomment for debugging
            continue

        # The last element in times should be the cumulative time of the last event
        # which corresponds to the total time of the sequence.
        total_time = times[-1]

        # Input sequence X: START, Event1, Event2, ... EventN
        x_seq = tokens[:-1]

        # Target cumulative times Y: Time1, Time2, ... TimeN
        # These are the cumulative times *at the end* of each step.
        y_seq = times[1:]

        # Mask: 1 for valid input tokens (not END_TOKEN), 0 otherwise
        # The mask applies to the *input* sequence (X_list).
        mask_seq = [1 if t != END_TOKEN else 0 for t in x_seq]

        X_list.append(x_seq)
        Y_list.append(y_seq)
        masks_list.append(mask_seq)
        total_times_list.append(total_time)

    if not X_list:
        print("No valid sequences found after preprocessing.")
        return np.array([]), np.array([]), np.array([]), np.array([])


    # Determine max length based on the processed sequences
    max_len = max(len(x) for x in X_list)
    print(f"Padding sequences to max length: {max_len}")

    # Pad sequences
    # X_train: pad with END_TOKEN (mask_zero=True in embedding will ignore this)
    X_train = pad_sequences(X_list, maxlen=max_len, padding='post', value=END_TOKEN)
    # Y_cum_target: pad with 0.0
    Y_cum_target = pad_sequences(Y_list, maxlen=max_len, padding='post', value=0.0)
    # mask_train: pad with 0
    mask_train = pad_sequences(masks_list, maxlen=max_len, padding='post', value=0)

    X_train = np.array(X_train, dtype=np.int32)
    Y_cum_target = np.array(Y_cum_target, dtype=np.float32)
    mask_train = np.array(mask_train, dtype=np.float32)
    total_times = np.array(total_times_list, dtype=np.float32)

    print(f"Prepared {X_train.shape[0]} sequences for training.")
    return X_train, Y_cum_target, mask_train, total_times

# %%
# ----------------------------
# Transformer Components (unchanged)
# ----------------------------
def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=16384, use_embedding=True):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = layers.Embedding(vocab_size, d_model, mask_zero=True)
        else:
            # If not using embedding, assume input is already dense (e.g., time features)
            self.embedding = layers.Dense(d_model, activation="relu")
        self.max_len = max_len
        # Ensure pos_encoding is created once and is large enough
        self.pos_encoding = positional_encoding(self.max_len, d_model)

    def compute_mask(self, x):
        # If using embedding with mask_zero, the mask is computed from the embedding layer
        if self.use_embedding:
             return self.embedding.compute_mask(x)
        # Otherwise, assume all steps are valid unless explicitly masked later
        return None # Or tf.math.not_equal(x, PAD_VALUE) if a pad value is used

    def call(self, x):
        # x is assumed to be token IDs if use_embedding is True, otherwise dense features
        if self.use_embedding:
            x = self.embedding(x)
        else:
             # Apply dense layer if input is not token IDs
             x = self.embedding(x)

        # Scale the embedding output
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add positional encoding
        seq_len = tf.shape(x)[1]
        # Ensure positional encoding slice matches sequence length
        x += self.pos_encoding[tf.newaxis, :seq_len, :]
        return x

class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        # Apply feed forward network with residual connection and layer normalization
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

class CausalSelfAttention(layers.Layer):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super().__init__()
        # MultiHeadAttention layer with causal mask
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        # Apply multi-head self-attention
        attn_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
        # Add residual connection and layer normalization
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        return x

class SelfAttentionFeedForwardLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        # Composes CausalSelfAttention and FeedForward layers
        self.self_attention = CausalSelfAttention(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x):
        # Pass input through self-attention and then feed-forward network
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1, max_len=16384):
        super().__init__()
        # Positional embedding for the input tokens
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model, max_len=max_len)
        # Stack of encoder layers
        self.enc_layers = [SelfAttentionFeedForwardLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        # Get the input mask from the embedding layer
        mask = self.pos_embedding.compute_mask(x)

        # Apply positional embedding and dropout
        x = self.pos_embedding(x)
        x = self.dropout(x)

        # Pass through encoder layers, applying the mask
        for layer in self.enc_layers:
            # Pass the mask to the encoder layers if they need it for attention
            # Note: CausalSelfAttention already uses use_causal_mask=True,
            # but the padding mask from embedding might also be needed.
            # Keras MHA handles mask from input automatically if compute_mask is implemented.
            x = layer(x) # Keras functional API handles mask propagation

        return x # The output tensor carries the mask

# Note: Decoder is not used in the TimeDiffTransformer as it's an encoder-only model
# class Decoder(tf.keras.Model):
#     ...

# %%
class TimeDiffTransformer(tf.keras.Model):
    """
    Transformer model predicting proportions of total time for each sequence step.
    This version removes the total time prediction head.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate=0.1, max_len=16384):
        super().__init__()
        # Encoder processes the input sequence of tokens
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate, max_len)

        # Head to predict the proportion of time for each step in the sequence
        # Output is a single value per sequence step before softmax
        self.proportion_head = layers.Dense(1)

    def call(self, inputs):
        # Pass input through the encoder
        encoder_out = self.encoder(inputs) # encoder_out shape: (batch_size, seq_len, d_model)
        # The mask from the embedding layer is propagated to encoder_out

        # Predict proportions for each step
        # proportions shape: (batch_size, seq_len, 1)
        proportions = self.proportion_head(encoder_out)

        # Remove the last dimension, proportions shape: (batch_size, seq_len)
        proportions = tf.squeeze(proportions, axis=-1)

        # Apply softmax across the sequence dimension to get proportions that sum to 1
        # The mask from the encoder_out should be used here to ignore padded values in softmax
        # Keras Softmax layer can handle masks, but explicit masking might be needed if custom logic is applied.
        # Let's rely on Keras mask propagation for now.
        proportions = tf.nn.softmax(proportions, axis=1)

        # Return only the predicted proportions
        return proportions # proportions shape: (batch_size, seq_len)

# %%
def compute_time_differences(proportions, total_time, mask):
    """
    Computes predicted increments and cumulative times from proportions and total time.
    Applies masking to ignore padded steps.

    Args:
        proportions: Predicted proportions for each step (batch_size, seq_len).
        total_time: The total time for each sequence (batch_size, 1).
        mask: Mask indicating valid steps (batch_size, seq_len).

    Returns:
        proportions: Normalized proportions (batch_size, seq_len).
        increments: Predicted time increments (batch_size, seq_len).
        cumulative_times: Predicted cumulative times (batch_size, seq_len).
    """
    # Ensure proportions and mask have the same shape
    # proportions = tf.reshape(proportions, tf.shape(mask)) # Removed redundant reshape

    # Apply mask to ensure only valid tokens contribute to calculations
    proportions *= tf.cast(mask, tf.float32)

    # Compute row-wise sum for normalization to handle variable-length sequences
    # Sum across the sequence length dimension (axis=1)
    row_sums = tf.reduce_sum(proportions, axis=1, keepdims=True)
    # Prevent division by zero if a sequence is entirely masked (shouldn't happen with START token)
    row_sums = tf.where(tf.equal(row_sums, 0), tf.ones_like(row_sums), row_sums)

    # Normalize proportions so they sum to 1 over the valid (unmasked) steps
    proportions /= row_sums

    # Compute increments by multiplying normalized proportions by the total time
    # total_time should have shape (batch_size, 1) for correct broadcasting
    increments = proportions * total_time # Broadcasting total_time

    # Compute cumulative times by summing increments along the sequence dimension
    cumulative_times = tf.math.cumsum(increments, axis=1)

    return proportions, increments, cumulative_times

# %%
def train_transformer(data_file, epochs=50, batch_size=32):
    """
    Trains the TimeDiffTransformer model.
    """
    try:
        # Load and preprocess data
        sequences_tokens, sequences_times, sequences_sourceids = load_and_preprocess_data(data_file)

        # Prepare data for training
        X_train, Y_cum_target, mask_train, total_times = prepare_training_data(sequences_tokens, sequences_times)

        if X_train.shape[0] == 0:
            print("No data available for training after preprocessing.")
            return None, None, None, None, None, None

        # Model parameters
        vocab_size = max(ENCODING_LEGEND.values()) + 1 # Include 0 for padding if not using mask_zero, but embedding handles it
        max_seq_len = X_train.shape[1]

        # Instantiate the model (now only predicts proportions)
        model = TimeDiffTransformer(
            num_layers=3,
            d_model=64,
            num_heads=8,
            dff=128,
            input_vocab_size=vocab_size,
            dropout_rate=0.1,
            max_len=max_seq_len # Pass max_len to the model
        )

        # Optimizer
        optimizer = tf.keras.optimizers.Adam()

        # Loss function for proportions (Mean Squared Error)
        # We will compute true proportions within the train step
        proportion_loss_fn = tf.keras.losses.MeanSquaredError()

        @tf.function
        def train_step(x, y_cum, mask): # Removed total_time from inputs
            with tf.GradientTape() as tape:
                # Model predicts proportions
                pred_props = model(x) # pred_props shape: (batch_size, seq_len)

                # Compute true time differences and total time from cumulative targets
                # time_diffs shape: (batch_size, seq_len - 1)
                time_diffs = y_cum[:, 1:] - y_cum[:, :-1]
                # true_total shape: (batch_size,)
                true_total = y_cum[:, -1] # Total time is the last cumulative time

                # Compute true proportions for the steps *after* the START token
                # true_total_expanded shape: (batch_size, 1)
                true_total_expanded = tf.where(
                    tf.equal(true_total, 0),
                    tf.ones_like(true_total),
                    true_total
                )[:, tf.newaxis]

                # true_props_unpadded shape: (batch_size, seq_len - 1) - corresponds to steps 1 to N
                true_props_unpadded = time_diffs / true_total_expanded

                # Pad true_props to match pred_props shape (batch_size, seq_len)
                # The first position (corresponding to START token input) should have 0 proportion
                true_props_padded = tf.pad(true_props_unpadded, [[0, 0], [1, 0]], constant_values=0.0)

                # Apply mask to both predicted and true proportions for loss calculation
                # Mask applies to the input sequence, which aligns with predicted proportions
                # Ensure mask is float32 for multiplication
                mask_float = tf.cast(mask, tf.float32)

                # Compute masked proportion loss
                # Only consider loss for steps where mask is 1
                masked_props_loss = proportion_loss_fn(true_props_padded * mask_float, pred_props * mask_float)

                total_loss = masked_props_loss # Total loss is just the proportion loss

            # Compute gradients and apply optimizer
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return total_loss, masked_props_loss

        # Training loop
        print("Starting training...")
        for epoch in range(epochs):
            # Pass only necessary data to train_step
            loss, props_loss = train_step(X_train, Y_cum_target, mask_train)
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {loss.numpy():.4f} - Proportion Loss: {props_loss.numpy():.4f}")

        print("Training finished.")
        return model, X_train, Y_cum_target, mask_train, total_times, sequences_sourceids

    except Exception as e:
        print(f"Error in train_transformer: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

# %%
def generate_predictions_csv(model, X_train, Y_cum_target, mask_train, total_times, sequences_sourceids):
    """
    Generates predictions using the trained model and saves them to a CSV file,
    correctly aligning SourceIDs with sequence steps.
    Uses the *true* total_times to compute predicted increments/cumulative times
    for comparison purposes, as the model no longer predicts total time.

    Args:
        model: The trained TimeDiffTransformer model (only predicts proportions).
        X_train: The input sequences (padded).
        Y_cum_target: The target cumulative times (padded).
        mask_train: The mask indicating valid sequence positions.
        total_times: The true total time for each sequence (NumPy array).
        sequences_sourceids: A list of lists, where each inner list contains
                             the original source IDs for a sequence.

    Returns:
        pandas.DataFrame: The DataFrame containing predictions and ground truth.
    """
    if model is None:
        print("Model is None, cannot generate predictions.")
        return pd.DataFrame()

    print("Generating predictions...")

    # Predict proportions using the model
    # model(X_train) returns proportions with shape (batch_size, seq_len)
    proportions_pred = model(X_train)

    # Compute predicted increments and cumulative times using the *true* total_times
    # Ensure total_times is a TensorFlow tensor with a batch dimension
    total_times_tf = tf.constant(total_times, dtype=tf.float32)
    # Expand total_times to shape (batch_size, 1) for broadcasting
    total_times_expanded = tf.expand_dims(total_times_tf, axis=1)

    # Use compute_time_differences with predicted proportions and true total time
    # compute_time_differences expects total_time with shape (batch_size, 1)
    proportions_pred_norm, increments_pred, cumulative_pred = compute_time_differences(
        proportions_pred, total_times_expanded, mask_train
    )

    # Convert TensorFlow tensors to NumPy arrays for easier handling
    proportions_pred_np = proportions_pred_norm.numpy()
    increments_pred_np = increments_pred.numpy()
    cumulative_pred_np = cumulative_pred.numpy()
    X_train_np = X_train # Already numpy
    Y_cum_target_np = Y_cum_target # Already numpy
    mask_train_np = mask_train # Already numpy

    # Compute ground truth increments for comparison
    # Handle the first element carefully (it's the time of the first event relative to start)
    gt_increments = np.zeros_like(Y_cum_target_np)
    # The first increment is the first cumulative time
    gt_increments[:, 0] = Y_cum_target_np[:, 0]
    # Subsequent increments are the differences between consecutive cumulative times
    gt_increments[:, 1:] = Y_cum_target_np[:, 1:] - Y_cum_target_np[:, :-1]
    # Apply mask to ground truth increments as well
    gt_increments *= mask_train_np


    # Collect predictions in a list of dictionaries for easy DataFrame creation
    output_records = []

    # Iterate through each sequence in the batch
    for seq_idx in range(X_train_np.shape[0]):
        # Find indices that are not padding (mask is 1)
        valid_mask = mask_train_np[seq_idx] == 1
        valid_indices = np.where(valid_mask)[0]

        # Get the original source IDs for this sequence
        # sequences_sourceids contains source IDs for steps *after* START token
        safe_sourceids = sequences_sourceids[seq_idx] if seq_idx < len(sequences_sourceids) else []

        step_counter = 1 # Initialize step counter for this sequence (starts from 1 for the first event)

        # Iterate through the valid indices within this sequence
        # valid_indices corresponds to positions in the padded sequence where mask is 1
        # valid_idx = 0 corresponds to the START token input
        # valid_idx = 1 corresponds to the first event input, etc.
        for i in range(len(valid_indices)):
            valid_idx = valid_indices[i] # The actual index in the padded sequence

            # We want to output predictions for each *event* step, not the START token step.
            # The predictions (proportions, increments, cumulative) at valid_idx
            # correspond to the time *until* the event at that position (relative to the previous event for increments,
            # or relative to the start for cumulative).
            # The SourceID at step 'k' (1-indexed) corresponds to the input token at index 'k' in the original sequence,
            # which is index 'k' in the padded input X_train, and index 'k-1' in the original sourceids list.
            # The predictions at index `valid_idx` relate to the event *after* the token at `valid_idx`.
            # So, prediction at `valid_idx` corresponds to the event with SourceID `safe_sourceids[valid_idx]`.

            # Check if this index corresponds to an actual event (not the START token)
            # The START token is at index 0 in the padded input.
            # The first event's input is at index 1, second at index 2, etc.
            # The predictions at index `j` correspond to the time *until* the event represented by the input token at index `j`.
            # So, pred at index `j` corresponds to SourceID at `safe_sourceids[j-1]`.
            # The valid_idx here is the index in the padded sequence.

            # Let's align predictions with the event they predict the time *until*.
            # Prediction at index `i` in padded sequence (where input is token `i`)
            # predicts time until event `i`. Event `i` has SourceID `safe_sourceids[i-1]`.
            # We should iterate through the *events*, which correspond to indices 1 onwards in the padded sequence.

            # Iterate through the valid event indices (skip the START token at index 0)
            if valid_idx > 0:
                # The source ID for the event predicted at valid_idx is the one at index valid_idx - 1
                source_id_index = valid_idx - 1

                # Get the corresponding source ID safely
                if source_id_index < len(safe_sourceids):
                    source_id = safe_sourceids[source_id_index]
                else:
                    # This case indicates a potential mismatch or issue elsewhere
                    source_id = f'Unknown_Mapping_Error_seq_{seq_idx}_idx_{source_id_index}'
                    print(f"Warning: Source ID index {source_id_index} out of bounds for sequence {seq_idx} with length {len(safe_sourceids)}")

                # Append record for this event step
                output_records.append({
                    'Sequence': seq_idx,
                    'Step': step_counter, # Use the dedicated counter for event steps
                    'SourceID': source_id,
                    'Predicted_Proportion': proportions_pred_np[seq_idx, valid_idx],
                    'Predicted_Increment': increments_pred_np[seq_idx, valid_idx],
                    'Predicted_Cumulative': cumulative_pred_np[seq_idx, valid_idx],
                    'GroundTruth_Increment': gt_increments[seq_idx, valid_idx],
                    'GroundTruth_Cumulative': Y_cum_target_np[seq_idx, valid_idx]
                })

                step_counter += 1 # Increment step counter only for actual events added

    # Create DataFrame from the collected records
    if not output_records:
        print("Warning: No valid prediction records generated.")
        predictions_df = pd.DataFrame(columns=[
            'Sequence', 'Step', 'SourceID', 'Predicted_Proportion',
            'Predicted_Increment', 'Predicted_Cumulative',
            'GroundTruth_Increment', 'GroundTruth_Cumulative'
        ])
    else:
        predictions_df = pd.DataFrame(output_records)

    # Save the DataFrame to CSV
    output_csv_path = 'predictions_transformer_175974.csv'
    try:
        predictions_df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved successfully to {output_csv_path}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")

    return predictions_df

# %%
def main():
    """
    Main function to run the training and prediction process.
    """
    try:
        # Replace with your actual file path
        data_file = "data/182625/encoded_182625_condensed.csv"
        # Check if the data file exists
        if not os.path.exists(data_file):
            print(f"Error: Data file not found at {data_file}")
            print("Please ensure the data file is in the correct location.")
            return

        # Train model and get results
        model, X_train, Y_cum_target, mask_train, total_times, sequences_sourceids = train_transformer(data_file)

        if model is None:
            print("Model training failed or no data was available. Exiting.")
            return

        # Generate predictions CSV
        predictions_df = generate_predictions_csv(
            model, X_train, Y_cum_target, mask_train, total_times, sequences_sourceids
        )

        if not predictions_df.empty:
            print("\nSample Predictions:")
            print(predictions_df.head(10))
        else:
            print("\nNo predictions were generated.")


    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure TensorFlow is using eager execution (usually default)
    # tf.config.run_functions_eagerly(True) # Uncomment for easier debugging if needed
    main()

