import pandas as pd
import os

# --- Configuration ---
# These are the correct relative paths for your local environment,
# assuming the script is run from your 'processing' folder.
INPUT_CSV_PATH = 'PXChange/data/176133/176133_raw.csv'
OUTPUT_CSV_PATH = 'PXChange/data/176133/encoded_176133.csv'

# Define the encoding legend for sourceID
SOURCEID_ENCODING_LEGEND = {
    'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7, 'MRI_FRR_34': 8, 'MRI_MPT_1005': 9,
    'MRI_MSR_100': 10, 'MRI_MSR_104': 11, 'MRI_MSR_21': 12,
    'START': 13,  # Start token
    'END': 14   # End Token
}

# Define the encoding legend for BodyGroup
BODYGROUP_ENCODING_LEGEND = {
    'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5, 'KNEE': 6,
    'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
}

# --- Load Data ---
try:
    # CRITICAL FIX 1: Read the CSV using a semicolon separator.
    data = pd.read_csv(INPUT_CSV_PATH, sep=';')
    print(f"✅ Successfully loaded data from {INPUT_CSV_PATH}")
    print(f"Initial data shape: {data.shape}")
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {INPUT_CSV_PATH}")
    print("Please ensure this script is in your 'processing' folder and the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- (Your analysis and mapping code remains the same) ---
# This section checks for values that are in your data but not in your legends.

print("\n--- Analyzing data before mapping ---")
unmapped_sourceIDs = set(data['sourceID'].dropna().unique()) - set(SOURCEID_ENCODING_LEGEND.keys())
if unmapped_sourceIDs:
    print(f"⚠️ Warning: Found {len(unmapped_sourceIDs)} 'sourceID' values in the CSV that are not in the legend. They will become 0.")

unmapped_bodygroup_from = set(data['BodyGroup_from'].dropna().unique()) - set(BODYGROUP_ENCODING_LEGEND.keys())
if unmapped_bodygroup_from:
    print(f"⚠️ Warning: Found {len(unmapped_bodygroup_from)} 'BodyGroup_from' values not in the legend. They will become 0.")

unmapped_bodygroup_to = set(data['BodyGroup_to'].dropna().unique()) - set(BODYGROUP_ENCODING_LEGEND.keys())
if unmapped_bodygroup_to:
     print(f"⚠️ Warning: Found {len(unmapped_bodygroup_to)} 'BodyGroup_to' values not in the legend. They will become 0.")


# --- Perform Mapping ---
print("\n--- Performing mapping and encoding ---")
data['sourceID'] = data['sourceID'].map(SOURCEID_ENCODING_LEGEND)
data['BodyGroup_from'] = data['BodyGroup_from'].map(BODYGROUP_ENCODING_LEGEND)
data['BodyGroup_to'] = data['BodyGroup_to'].map(BODYGROUP_ENCODING_LEGEND)

# Fill any values that weren't in the legends (resulting in NaN) with 0
data.fillna(0, inplace=True)

# Ensure the columns are integer type after filling
data['sourceID'] = data['sourceID'].astype(int)
data['BodyGroup_from'] = data['BodyGroup_from'].astype(int)
data['BodyGroup_to'] = data['BodyGroup_to'].astype(int)

# --- Save Data ---
print("\n--- Saving encoded data ---")
try:
    # Ensure the output directory exists before saving
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # CRITICAL FIX 2: Save the new CSV. The default separator is a comma.
    data.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"✅ Success! Encoded data saved to {OUTPUT_CSV_PATH}")
    print(f"Final data shape: {data.shape}")

except Exception as e:
    print(f"❌ Error saving file: {e}")