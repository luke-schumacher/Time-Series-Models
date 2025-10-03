import pandas as pd
import sys
import os

# --- 1. Setup File Paths ---
# The script will read from the raw data file you provided.
input_file_path = 'PXChange/data/176401/encoded_176401.csv'
# The final, processed data will be saved to this new file.
output_file_path = 'PXChange/data/176401/encoded_176401_condensed.csv'


# --- 2. Define Final Column Structure ---
# This list defines the exact columns and their order in the final output file.
# The original 'Position' and 'Direction' columns have been removed.
final_columns_to_keep = [
    'SeqOrder',
    'sourceID',
    'timediff',
    'PTAB',
    'BodyGroup_from',
    'BodyGroup_to',
    'PatientID_from',
    'PatientID_to',
    'Age',
    'Weight',
    'Height',
    'Position_encoded',
    'Direction_encoded'
]


# --- 3. Main Processing Logic ---
try:
    # Load the source CSV file.
    # Note: The 'sourceID' column in the raw CSV is sometimes read as text,
    # so we explicitly handle potential conversion errors.
    df = pd.read_csv(input_file_path)
    df['sourceID'] = pd.to_numeric(df['sourceID'], errors='coerce').fillna(0).astype(int)
    print(f"✅ Successfully loaded '{input_file_path}'.")

    # --- Create the 'SeqOrder' column (Corrected Logic) ---
    # This logic identifies each time 'sourceID' is 11 as the start of a new group.
    # .cumsum() on the boolean series creates a unique ID for each group.
    # We subtract 1 to make the sequence numbering start from 0.
    is_sequence_start = (df['sourceID'] == 11)
    df['SeqOrder'] = is_sequence_start.cumsum() - 1
    print(" -> 'SeqOrder' column generated with correct grouping logic.")

    # --- Encode Categorical Columns ---
    # Creates new columns with numerical representations of 'Position' and 'Direction'.
    if 'Position' in df.columns:
        df['Position_encoded'] = pd.factorize(df['Position'])[0]
    if 'Direction' in df.columns:
        df['Direction_encoded'] = pd.factorize(df['Direction'])[0]
    print(" -> 'Position' and 'Direction' columns encoded.")

    # --- Filter and Reorder Columns ---
    # Selects only the columns defined in the 'final_columns_to_keep' list
    # and drops all others. This ensures the final CSV is clean.
    df_final = df[final_columns_to_keep]
    print(" -> Columns filtered and reordered to final specification.")

    # --- Save the Result to a New File ---
    # Create the output directory if it doesn't exist to prevent errors.
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Saves the processed data to the new CSV file without an index column.
    df_final.to_csv(output_file_path, index=False)
    print(f"\n🎉 Success! Processed file saved as: {output_file_path}")

except FileNotFoundError:
    print(f"❌ Error: The file '{input_file_path}' was not found.")
    print("Please make sure the script and the CSV file are in the same folder.")

except KeyError as e:
    print(f"❌ Error: A required column is missing from the CSV: {e}")
    print("Please check that your input CSV contains all necessary columns and that the names match exactly (including capitalization).")

