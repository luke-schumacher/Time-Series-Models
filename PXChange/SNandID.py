import pandas as pd

# Define file name
file_name = "PXChange/182625_with_timediff.csv"
output_file = "PXChange/182625_with_timediff_updated.csv"

# Load the CSV file
df = pd.read_csv(file_name)

# Add SN column with constant value 182625
df['SN'] = 182625

# Add ID column with incrementing values
df.insert(0, 'ID', range(1, len(df) + 1))

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")
