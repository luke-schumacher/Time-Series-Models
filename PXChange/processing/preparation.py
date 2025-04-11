import pandas as pd

# Path to your existing CSV file (replace with your file path)
file_path = 'PXChange/encoded_175651_condensed.csv'  # Update with your file path

# Load the CSV file
df = pd.read_csv(file_path)

# Remove the 'order_id' column if it exists
df = df.drop(columns=['order_id'], errors='ignore')

# Remove the 'SeqOrder' column if it already exists (to avoid duplicates)
df = df.drop(columns=['SeqOrder'], errors='ignore')

# Initialize a list to store SeqOrder values
seq_order = []

# Initialize a counter and a variable to track the last sourceID
counter = 0
last_sourceID = None

# Iterate through each row and assign SeqOrder
for sourceID in df['sourceID']:
    if sourceID == last_sourceID or last_sourceID is None:
        # Continue counting if the sourceID is the same
        seq_order.append(counter)
        counter += 1
    else:
        # Reset the counter when a new sourceID (other than 10) is encountered
        if sourceID == 10:
            counter = 0  # Reset to 0 for sourceID 10
        seq_order.append(counter)
        counter += 1
    last_sourceID = sourceID

# Add the SeqOrder column to the DataFrame (at the front)
df.insert(0, 'SeqOrder', seq_order)

# Save the updated DataFrame back to the same CSV file (overwrite the existing file)
df.to_csv(file_path, index=False)

print(f"Updated CSV file saved to {file_path}")


