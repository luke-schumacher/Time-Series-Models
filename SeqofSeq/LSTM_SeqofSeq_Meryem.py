import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses all INFO-level messages
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K  # For custom loss function

# 1. Load and prepare data
file_path = 'prepared_data_176625.csv'
data = pd.read_csv(file_path)

# Flatten the 'Durations' column to extract all available duration values per patient
flattened_data = []
for index, row in data.iterrows():
    patient_id = row['PatientID']
    durations = eval(row['Durations'])  # Convert string representation of lists to actual lists
    for i, duration in enumerate(durations):
        flattened_data.append({
            'PatientID': patient_id,
            'TimeStep': i,
            'Duration': duration[0]  # We figured out that the padding which we added to the durations were causing the problem. The new data does not have any padding anymore, the shape is different. 
        })

# Create a DataFrame from the flattened data
time_series_data = pd.DataFrame(flattened_data)

# Get maximum sequence length across patients
max_sequence_length = time_series_data.groupby('PatientID').size().max()
print(f"Detected maximum sequence length: {max_sequence_length}")

# 2. Prepare sequences and targets for multi-step prediction
sequence_length = max_sequence_length
prediction_length = 10  # Modify this if you only want to predict a few steps ahead

sequences, targets, patient_ids = [], [], []
for i in range(len(time_series_data) - sequence_length - prediction_length + 1):
    seq = time_series_data['Duration'].iloc[i:i + sequence_length].values
    target = time_series_data['Duration'].iloc[i + sequence_length: i + sequence_length + prediction_length].values
    patient_id = time_series_data['PatientID'].iloc[i]  # Capture the patient ID
    sequences.append(seq)
    targets.append(target)
    patient_ids.append(patient_id)

sequences = np.array(sequences)
targets = np.array(targets)
patient_ids = np.array(patient_ids)

# 3. Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Ensure scaling is non-negative
sequences = scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
targets = scaler.transform(targets.reshape(-1, 1)).reshape(targets.shape)

# 4. Reshape data for LSTM (samples, timesteps, features)
sequences = sequences.reshape(sequences.shape[0], sequence_length, 1)  # Add print here to confirm shape change

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test, train_patient_ids, test_patient_ids = train_test_split(
    sequences, targets, patient_ids, test_size=0.2, random_state=42)

# 6. Implement cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 7. Build the LSTM model with added regularization and early stopping
def create_model():
    inputs = Input(shape=(sequence_length, 1)) 
    x = LSTM(64, return_sequences=True, kernel_regularizer='l2')(inputs)  # Modify LSTM parameters here for batched processing
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LSTM(32, kernel_regularizer='l2')(x)  # Modify LSTM parameters here for batched processing
    x = Dropout(0.3)(x)
    outputs = Dense(prediction_length, activation='relu')(x)  # Output with ReLU activation to ensure non-negativity
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Custom loss function to penalize negative predictions
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true) + K.square(K.minimum(y_pred, 0)))

# 8. Compile the model
model = create_model()
model.compile(optimizer='adam', loss=custom_loss)

# 9. Implement early stopping with a reduced patience to avoid excessive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with cross-validation and track the history for visualization
fold_no = 1
histories = []
for train_index, val_index in kf.split(X_train):
    print(f"Training on fold {fold_no}")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=20,  # Reduced epochs
        batch_size=16,  # Reduced batch size
        callbacks=[early_stopping],
        verbose=1
    )
   
    histories.append(history)
    fold_no += 1

# Plot training and validation loss for each fold
plt.figure(figsize=(10, 6))
for fold, history in enumerate(histories, 1):
    plt.plot(history.history['loss'], label=f'Training Loss Fold {fold}')
    plt.plot(history.history['val_loss'], '--', label=f'Validation Loss Fold {fold}')
plt.title('Training and Validation Loss across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 10. Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# 11. Make future predictions
predictions = model.predict(X_test)  # Add print to check predictions shape

# Reverse scaling to original values for interpretation and clip predictions to non-negative
y_test_unscaled = scaler.inverse_transform(y_test)
predictions_unscaled = np.clip(scaler.inverse_transform(predictions), a_min=0, a_max=None)

# 12. Save predictions and actual values to CSV for easier comparison
results_df = pd.DataFrame()
steps_to_plot = min(prediction_length, 3)  # Limit the steps to plot for clarity
for step in range(steps_to_plot):
    results_df[f'Actual Step {step+1}'] = y_test_unscaled[:, step]
    results_df[f'Predicted Step {step+1}'] = predictions_unscaled[:, step]

# Save to CSV for review
results_df.to_csv("predictions_comparison.csv", index=False)

# 13. Visualize actual vs predicted values for a sample of the test set
plt.figure(figsize=(12, 6))
for i in range(steps_to_plot):
    plt.plot(results_df[f'Actual Step {i+1}'], label=f'Actual Step {i+1}', marker='o')
    plt.plot(results_df[f'Predicted Step {i+1}'], label=f'Predicted Step {i+1}', linestyle='--', marker='x')
plt.title('Actual vs. Predicted Durations (Limited Steps)')
plt.xlabel('Sample')
plt.ylabel('Duration')
plt.legend()
plt.show()

# 14. Track Actual and Predicted values for each time step
stepwise_results = []
for i in range(len(X_test)):
    patient_id = test_patient_ids[i]  # Get the corresponding patient ID for this sequence
    actual_values = y_test_unscaled[i]
    predicted_values = predictions_unscaled[i]

    # For each time step, print and store the actual vs predicted values
    for t in range(prediction_length):
        stepwise_results.append({
            'PatientID': patient_id,
            'TimeStep': t + 1,
            'Actual': actual_values[t],
            'Predicted': predicted_values[t]
        })

# Convert to DataFrame for better readability
stepwise_results_df = pd.DataFrame(stepwise_results)

# Save the detailed results to a CSV for review
stepwise_results_df.to_csv("stepwise_predictions_comparison.csv", index=False)
