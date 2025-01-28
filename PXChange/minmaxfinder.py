import pandas as pd

# Path to your CSV file
input_file = 'filtered_182625.csv'  # Replace with your file path

# Load the dataset
data = pd.read_csv(input_file)

# Check if the column exists
if 'timediff' not in data.columns:
    raise ValueError("The column 'timediff' does not exist in the dataset.")

# Find the min and max values of the 'timediff' column
min_value = data['timediff'].min()
max_value = data['timediff'].max()

print(f"Minimum value in 'timediff': {min_value}")
print(f"Maximum value in 'timediff': {max_value}")
