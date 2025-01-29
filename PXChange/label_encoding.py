import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the encoding legend for sourceID
SOURCEID_ENCODING_LEGEND = {
    'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_3': 6, 'MRI_FRR_34': 7, 'MRI_MPT_1005': 8,
    'MRI_MSR_100': 9, 'MRI_MSR_104': 10, 'MRI_MSR_21': 11, 'MRI_MSR_34': 12,
    'START': 13,  # Start token
    'END': 14     # End token
}

# Define the encoding legend for BodyGroup
BODYGROUP_ENCODING_LEGEND = {
    'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5, 'KNEE': 6,
    'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
}

# Load your dataset
data = pd.read_csv('PXChange/Time_difference_gen_data.csv')

# Map the encoding legends to the relevant columns
data['sourceID'] = data['sourceID'].map(SOURCEID_ENCODING_LEGEND)
data['BodyGroup_from'] = data['BodyGroup_from'].map(BODYGROUP_ENCODING_LEGEND)
data['BodyGroup_to'] = data['BodyGroup_to'].map(BODYGROUP_ENCODING_LEGEND)

# Check for unmapped values and handle them (optional)
if data[['sourceID', 'BodyGroup_from', 'BodyGroup_to']].isnull().any().any():
    print("Warning: Some values were not found in the encoding legends.")
    # Handle unmapped values (e.g., fill with 0 or another default value)
    data.fillna(0, inplace=True)

# Normalize numerical columns
scaler = MinMaxScaler()
data[['PTAB']] = scaler.fit_transform(data[['PTAB']])

# Save the normalized, encoded data to a CSV file
output_path = "encoded_normalized_data.csv"
data.to_csv(output_path, index=False)

print(f"Encoded and normalized data saved to {output_path}")
