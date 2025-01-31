import pandas as pd

# Define the original min and max values used for normalization
original_min = 0  # Replace with the actual minimum value
original_max = 2900  # Replace with the actual maximum value

# Load your dataset
input_file = 'PXChange/182625_with_timediff.csv'  # Replace with your file path

data = pd.read_csv(input_file)

# Ensure the column exists
if 'Predicted_timediff' not in data.columns:
    raise ValueError("The column 'Predicted_timediff' does not exist in the dataset.")

# Perform denormalization
data['Denormalized Values'] = data['Predicted_timediff'] * (original_max - original_min) + original_min

# Save the updated dataset with the denormalized column to the same file
data.to_csv(input_file, index=False)

print(f"Denormalized values added and saved to {input_file}")
