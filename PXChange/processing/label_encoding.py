import pandas as pd
# Note: MinMaxScaler is imported in your original code but not used.
# If you need normalization, you'll need to add that step back in.
# from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
INPUT_CSV_PATH = 'PXCHange/processing/filtered_175651.csv'
OUTPUT_CSV_PATH = "PXChange/encoded_175651.csv" # Changed output name slightly

# Define the encoding legend for sourceID
SOURCEID_ENCODING_LEGEND = {
    'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7, 'MRI_FRR_34': 8, 'MRI_MPT_1005': 9,
    'MRI_MSR_100': 10, 'MRI_MSR_104': 11, 'MRI_MSR_21': 12, 'MRI_MSR_34': 13,
    'START': 14,  # Start token
    'END': 15  # End Token
}

# Define the encoding legend for BodyGroup
BODYGROUP_ENCODING_LEGEND = {
    'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5, 'KNEE': 6,
    'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
}

# --- Load Data ---
try:
    data = pd.read_csv(INPUT_CSV_PATH)
    print(f"Successfully loaded data from {INPUT_CSV_PATH}")
    print(f"Initial data shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit() # Exit if the file doesn't exist

# --- Pre-Mapping Analysis ---

print("\n--- Analyzing 'sourceID' before mapping ---")
# Get unique values present in the data's sourceID column
original_sourceIDs = data['sourceID'].unique()
print(f"Found {len(original_sourceIDs)} unique values in 'sourceID' column.")

# Get the keys defined in the legend
legend_sourceID_keys = set(SOURCEID_ENCODING_LEGEND.keys())

# Find which original values are NOT in the legend keys
unmapped_sourceIDs = set(original_sourceIDs) - legend_sourceID_keys

if unmapped_sourceIDs:
    print("\n*** WARNING: The following 'sourceID' values exist in the CSV but are NOT in the SOURCEID_ENCODING_LEGEND: ***")
    for val in sorted(list(unmapped_sourceIDs)): # Sort for easier reading
        count = (data['sourceID'] == val).sum()
        print(f"  - '{val}' (appears {count} times)")
    print("*** These values will become NaN after mapping and then 0 after fillna(0). ***")
else:
    print("All 'sourceID' values in the CSV are present in the updated encoding legend.") # Updated message slightly

# Optional: Show value counts before mapping (can be long if many unique values)
# print("\nValue counts for 'sourceID' BEFORE mapping:")
# print(data['sourceID'].value_counts(dropna=False)) # dropna=False shows count of NaNs if any existed initially


print("\n--- Analyzing 'BodyGroup_from' before mapping ---")
original_bodygroup_from = data['BodyGroup_from'].unique()
legend_bodygroup_keys = set(BODYGROUP_ENCODING_LEGEND.keys())
unmapped_bodygroup_from = set(original_bodygroup_from) - legend_bodygroup_keys
if unmapped_bodygroup_from:
    print("\n*** WARNING: The following 'BodyGroup_from' values exist in the CSV but are NOT in the BODYGROUP_ENCODING_LEGEND: ***")
    for val in sorted(list(unmapped_bodygroup_from)):
         count = (data['BodyGroup_from'] == val).sum()
         print(f"  - '{val}' (appears {count} times)")
    print("*** These values will become NaN after mapping and then 0 after fillna(0). ***")
else:
    print("All 'BodyGroup_from' values in the CSV are present in the encoding legend.")

print("\n--- Analyzing 'BodyGroup_to' before mapping ---")
original_bodygroup_to = data['BodyGroup_to'].unique()
# legend_bodygroup_keys is the same as for 'BodyGroup_from'
unmapped_bodygroup_to = set(original_bodygroup_to) - legend_bodygroup_keys
if unmapped_bodygroup_to:
    print("\n*** WARNING: The following 'BodyGroup_to' values exist in the CSV but are NOT in the BODYGROUP_ENCODING_LEGEND: ***")
    for val in sorted(list(unmapped_bodygroup_to)):
         count = (data['BodyGroup_to'] == val).sum()
         print(f"  - '{val}' (appears {count} times)")
    print("*** These values will become NaN after mapping and then 0 after fillna(0). ***")
else:
    print("All 'BodyGroup_to' values in the CSV are present in the encoding legend.")


# --- Perform Mapping ---
print("\n--- Performing mapping ---")
data['sourceID'] = data['sourceID'].map(SOURCEID_ENCODING_LEGEND)
data['BodyGroup_from'] = data['BodyGroup_from'].map(BODYGROUP_ENCODING_LEGEND)
data['BodyGroup_to'] = data['BodyGroup_to'].map(BODYGROUP_ENCODING_LEGEND)

# --- Post-Mapping Check (Before fillna) ---
print("\n--- Checking for NaN values after mapping (before fillna) ---")
nan_check = data[['sourceID', 'BodyGroup_from', 'BodyGroup_to']].isnull()
if nan_check.any().any():
    print("NaN values found after mapping, likely due to unmapped original values in BodyGroup columns or remaining unmapped sourceIDs.") # Adjusted message
    print("Number of NaN values per column:")
    print(nan_check.sum())

    # Optional: Show value counts including NaN *after* mapping
    print("\nValue counts for 'sourceID' AFTER mapping (including NaN):")
    print(data['sourceID'].value_counts(dropna=False))
    print("\nValue counts for 'BodyGroup_from' AFTER mapping (including NaN):")
    print(data['BodyGroup_from'].value_counts(dropna=False))
    print("\nValue counts for 'BodyGroup_to' AFTER mapping (including NaN):")
    print(data['BodyGroup_to'].value_counts(dropna=False))

    # Handle unmapped values by filling with 0
    print("\nFilling NaN values with 0...")
    data.fillna(0, inplace=True)

    # --- Post-fillna Check ---
    print("\nValue counts for 'sourceID' AFTER mapping and fillna(0):")
    print(data['sourceID'].value_counts(dropna=False)) # Should now show 0 count instead of NaN
    print("\nValue counts for 'BodyGroup_from' AFTER mapping and fillna(0):")
    print(data['BodyGroup_from'].value_counts(dropna=False))
    print("\nValue counts for 'BodyGroup_to' AFTER mapping and fillna(0):")
    print(data['BodyGroup_to'].value_counts(dropna=False))

else:
    print("No NaN values found after mapping. All original values were successfully mapped with the updated legends.") # Updated message


# --- Save Data ---
print("\n--- Saving encoded data ---")
# Ensure columns are integer type if desired (fillna might make them float)
# This is important if subsequent code expects integers
data['sourceID'] = data['sourceID'].astype(int)
data['BodyGroup_from'] = data['BodyGroup_from'].astype(int)
data['BodyGroup_to'] = data['BodyGroup_to'].astype(int)

data.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nEncoded data saved to {OUTPUT_CSV_PATH}")
print(f"Final data shape: {data.shape}")