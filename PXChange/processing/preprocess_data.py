import pandas as pd
import os
import re

def preprocess_176401_data(input_raw_csv_path, output_encoded_csv_path):
    """
    Combines label encoding, sequence ordering, and feature encoding for the 176401 dataset.
    Also finds min/max for 'timediff' for normalization purposes.

    Args:
        input_raw_csv_path (str): Absolute path to the raw 176401 CSV file (e.g., '.../176401_raw_full.csv').
        output_encoded_csv_path (str): Absolute path where the preprocessed CSV will be saved.

    Returns:
        tuple: (pd.DataFrame, float, float) - The preprocessed DataFrame, min_timediff, max_timediff.
    """

    # --- Configuration for Encoding Legends ---
    SOURCEID_ENCODING_LEGEND = {
        'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
        'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7, 'MRI_FRR_34': 8, 'MRI_MPT_1005': 9,
        'MRI_MSR_100': 10, 'MRI_MSR_104': 11, 'MRI_MSR_21': 12,
        'START': 13,  # Start token
        'END': 14   # End Token
    }

    BODYGROUP_ENCODING_LEGEND = {
        'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5, 'KNEE': 6,
        'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
    }

    # --- 1. Load Data ---
    try:
        # Assuming raw data might use ';' as separator based on label_encoding.py
        # Try comma first, then semicolon
        try:
            df = pd.read_csv(input_raw_csv_path)
        except Exception:
            df = pd.read_csv(input_raw_csv_path, sep=';')
        
        print(f"✅ Successfully loaded data from {input_raw_csv_path}")
        print(f"Initial data shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_raw_csv_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None, None

    # --- 2. Label Encoding (from label_encoding.py) ---
    print("--- Performing label encoding ---")
    # Handle potential missing columns gracefully
    if 'sourceID' in df.columns:
        df['sourceID'] = df['sourceID'].map(SOURCEID_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        print("Warning: 'sourceID' column not found for label encoding.")
        df['sourceID'] = 0 # Add a default column if missing

    if 'BodyGroup_from' in df.columns:
        df['BodyGroup_from'] = df['BodyGroup_from'].map(BODYGROUP_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        print("Warning: 'BodyGroup_from' column not found for label encoding.")
        df['BodyGroup_from'] = 0

    if 'BodyGroup_to' in df.columns:
        df['BodyGroup_to'] = df['BodyGroup_to'].map(BODYGROUP_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        print("Warning: 'BodyGroup_to' column not found for label encoding.")
        df['BodyGroup_to'] = 0
    
    print(" -> 'sourceID', 'BodyGroup_from', 'BodyGroup_to' encoded.")

    # --- 3. Create 'SeqOrder' column (from preparation.py) ---
    print("--- Generating 'SeqOrder' column ---")
    # Ensure 'sourceID' is numeric for this operation
    df['sourceID'] = pd.to_numeric(df['sourceID'], errors='coerce').fillna(0).astype(int)
    is_sequence_start = (df['sourceID'] == 11) # Assuming 11 is the encoded value for 'MRI_MSR_104' or similar start token
    df['SeqOrder'] = is_sequence_start.cumsum() - 1
    print(" -> 'SeqOrder' column generated.")

    # --- 4. Encode Categorical Columns 'Position' and 'Direction' (from preparation.py) ---
    print("--- Encoding 'Position' and 'Direction' ---")
    if 'Position' in df.columns:
        df['Position_encoded'] = pd.factorize(df['Position'])[0]
    else:
        print("Warning: 'Position' column not found for encoding.")
        df['Position_encoded'] = 0

    if 'Direction' in df.columns:
        df['Direction_encoded'] = pd.factorize(df['Direction'])[0]
    else:
        print("Warning: 'Direction' column not found for encoding.")
        df['Direction_encoded'] = 0
    print(" -> 'Position' and 'Direction' columns encoded.")

    # --- 5. Find Min/Max for 'timediff' (from minmaxfinder.py) ---
    min_timediff = None
    max_timediff = None
    if 'timediff' in df.columns:
        min_timediff = df['timediff'].min()
        max_timediff = df['timediff'].max()
        print(f"Min 'timediff': {min_timediff}, Max 'timediff': {max_timediff}")
    else:
        print("Warning: 'timediff' column not found. Cannot determine min/max for normalization.")

    # --- 6. Filter and Reorder Columns (from preparation.py) ---
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
    
    # Filter to only include columns that actually exist in the DataFrame
    existing_columns_to_keep = [col for col in final_columns_to_keep if col in df.columns]
    df_final = df[existing_columns_to_keep]
    print(" -> Columns filtered and reordered to final specification.")

    # --- 7. Save the Result ---
    output_dir = os.path.dirname(output_encoded_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_final.to_csv(output_encoded_csv_path, index=False)
    print(f"🎉 Success! Preprocessed file saved as: {output_encoded_csv_path}")

    return df_final, min_timediff, max_timediff

if __name__ == "__main__":
    # Example usage:
    # Adjust these paths as necessary for testing
    input_path = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data/176401/176401_raw_full.csv'
    output_path = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data/176401/preprocessed_176401.csv'
    
    processed_df, min_val, max_val = preprocess_176401_data(input_path, output_path)
    
    if processed_df is not None:
        print("First 5 rows of the preprocessed data:")
        print(processed_df.head())
        print(f"Min timediff: {min_val}, Max timediff: {max_val}")
