import pandas as pd

# Define the original min and max values used for normalization
original_min = 0  # Replace with the actual minimum value
original_max = 2900  # Replace with the actual maximum value

# Load your dataset
input_file = 'dataset_normalized_predictions.csv'  # Replace with your file path
output_file = 'timemdiff_data.csv'  # Path to save the denormalized data

data = pd.read_csv(input_file)

# Ensure the column exists
if 'Predicted Values' not in data.columns:
    raise ValueError("The column 'Predicted Values' does not exist in the dataset.")

# Perform denormalization
data['Denormalized Values'] = data['Predicted Values'] * (original_max - original_min) + original_min

# Save the new dataset with the denormalized column to a new CSV file
data[['Denormalized Values']].to_csv(output_file, index=False)

print(f"Denormalized values saved to {output_file}")
