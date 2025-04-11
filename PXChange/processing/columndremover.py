import pandas as pd

# Path to the input CSV file
input_file = 'PXChange/encoded_182625_debugged.csv'  # Replace with your file path
output_file = 'encoded_182625_debugged_condensed.csv'  # File to save the filtered dataset

# Specify the columns to keep
columns_to_keep = ['sourceID', 'timediff', 'PTAB', 'BodyGroup_from', 'BodyGroup_to']

# Load the dataset
df = pd.read_csv(input_file)

# Filter the dataset to include only the specified columns
filtered_df = df[columns_to_keep]

# Save the filtered dataset to a new CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to {output_file}")
