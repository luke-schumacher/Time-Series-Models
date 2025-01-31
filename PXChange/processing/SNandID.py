import pandas as pd

# Define file names
file_name = "PXChange/new_data_with_reset_predictions_2.csv"
output_file = "PXChange/decoded_182625_final.csv"

# Decoding legend for sourceID
DECODING_LEGEND = {v: k for k, v in {
    'Not Vital': 0, 'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_3': 6, 'MRI_FRR_34': 7, 'MRI_MPT_1005': 8,
    'MRI_MSR_100': 9, 'MRI_MSR_104': 10, 'MRI_MSR_21': 11, 'MRI_MSR_34': 12
}.items()}

# Decoding legend for bodygroup_from and bodygroup_to
BODYGROUP_DECODING = {v: k for k, v in {
    'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5,
    'KNEE': 6, 'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
}.items()}

# Load the CSV file
df = pd.read_csv(file_name)

# Decode the sourceID column
df['sourceID_encoded'] = df['sourceID_encoded'].map(DECODING_LEGEND)

# Decode the bodygroup_from and bodygroup_to columns
df['BodyGroup_from_encoded'] = df['BodyGroup_from_encoded'].map(BODYGROUP_DECODING)
df['BodyGroup_to_encoded'] = df['BodyGroup_to_encoded'].map(BODYGROUP_DECODING)

# Remove the orderID and PTAB columns
df = df.drop(columns=['PTAB'], errors='ignore')

# Add SN column with constant value 182625
df['SN'] = 182625

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated file saved as {output_file}")

