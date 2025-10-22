import pandas as pd
import os
import glob

def preprocess_data(dataset_id, base_data_dir='C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data'):
    """
    Combines label encoding, sequence ordering, and feature encoding for a given dataset.
    Also finds min/max for 'timediff' for normalization purposes.

    Args:
        dataset_id (str): The ID of the dataset (e.g., '176401').
        base_data_dir (str): The base directory where data is stored.

    Returns:
        tuple: (pd.DataFrame, float, float) - The preprocessed DataFrame, min_timediff, max_timediff.
    """
    input_dir = os.path.join(base_data_dir, dataset_id)
    
    # Find any CSV file that is not a derivative file
    potential_files = glob.glob(os.path.join(input_dir, '*.csv'))
    raw_files = [f for f in potential_files if not any(name in os.path.basename(f) for name in ['encoded', 'filtered', 'predictions', 'preprocessed'])]

    if not raw_files:
        print(f"❌ Error: No suitable raw data file found for dataset {dataset_id} in {input_dir}")
        return None, None, None
    
    input_raw_csv_path = raw_files[0] # Take the first suitable file found
    
    output_encoded_csv_path = os.path.join(input_dir, f'preprocessed_{dataset_id}.csv')

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
        try:
            df = pd.read_csv(input_raw_csv_path)
        except Exception:
            df = pd.read_csv(input_raw_csv_path, sep=';')
        
        print(f"✅ Successfully loaded data for dataset {dataset_id} from {input_raw_csv_path}")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_raw_csv_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None, None

    # --- 2. Label Encoding ---
    if 'sourceID' in df.columns:
        df['sourceID'] = df['sourceID'].map(SOURCEID_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        df['sourceID'] = 0

    if 'BodyGroup_from' in df.columns:
        df['BodyGroup_from'] = df['BodyGroup_from'].map(BODYGROUP_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        df['BodyGroup_from'] = 0

    if 'BodyGroup_to' in df.columns:
        df['BodyGroup_to'] = df['BodyGroup_to'].map(BODYGROUP_ENCODING_LEGEND).fillna(0).astype(int)
    else:
        df['BodyGroup_to'] = 0

    # --- 3. Create 'SeqOrder' column ---
    df['sourceID'] = pd.to_numeric(df['sourceID'], errors='coerce').fillna(0).astype(int)
    is_sequence_start = (df['sourceID'] == 11)
    df['SeqOrder'] = is_sequence_start.cumsum() - 1

    # --- 4. Encode Categorical Columns 'Position' and 'Direction' ---
    if 'Position' in df.columns:
        df['Position_encoded'] = pd.factorize(df['Position'])[0]
    else:
        df['Position_encoded'] = 0

    if 'Direction' in df.columns:
        df['Direction_encoded'] = pd.factorize(df['Direction'])[0]
    else:
        df['Direction_encoded'] = 0

    # --- 5. Find Min/Max for 'timediff' ---
    min_timediff, max_timediff = None, None
    if 'timediff' in df.columns:
        min_timediff = df['timediff'].min()
        max_timediff = df['timediff'].max()

    # --- 6. Filter and Reorder Columns ---
    final_columns_to_keep = [
        'SeqOrder', 'sourceID', 'timediff', 'PTAB', 'BodyGroup_from', 'BodyGroup_to',
        'PatientID_from', 'PatientID_to', 'Age', 'Weight', 'Height',
        'Position_encoded', 'Direction_encoded'
    ]
    
    existing_columns_to_keep = [col for col in final_columns_to_keep if col in df.columns]
    df_final = df[existing_columns_to_keep]

    # --- 7. Save the Result ---
    df_final.to_csv(output_encoded_csv_path, index=False)
    print(f"🎉 Success! Preprocessed file for dataset {dataset_id} saved as: {output_encoded_csv_path}")

    return df_final, min_timediff, max_timediff

def main():
    """
    Finds all dataset directories in the 'data' folder and preprocesses each one.
    """
    base_data_dir = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange/data'
    
    dataset_ids = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    
    print(f"Found {len(dataset_ids)} potential datasets: {dataset_ids}")
    
    for dataset_id in dataset_ids:
        print(f"\n--- Processing dataset: {dataset_id} ---")
        preprocess_data(dataset_id, base_data_dir)

if __name__ == "__main__":
    main()