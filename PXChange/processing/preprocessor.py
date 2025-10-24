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
    
    # Find all CSV files in the directory
    all_csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    # Exclude files that are clearly processed/derived
    raw_candidates = []
    for f in all_csv_files:
        basename = os.path.basename(f).lower()
        if not any(name in basename for name in ['encoded', 'filtered', 'predictions', 'preprocessed']):
            raw_candidates.append(f)

    input_raw_csv_path = None
    if raw_candidates:
        # Take the largest non-derived file as the raw data
        input_raw_csv_path = max(raw_candidates, key=os.path.getsize)

    if not input_raw_csv_path:
        print(f"❌ Error: No suitable raw data file found for dataset {dataset_id} in {input_dir}")
        return None, None, None
    
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

    # --- 5. Calculate true_total_time and filter ---
    min_timediff, max_timediff = None, None
    
    # Check for 'timediff' or 'Predicted_timediff'
    timediff_col = None
    if 'timediff' in df.columns:
        timediff_col = 'timediff'
    elif 'Predicted_timediff' in df.columns:
        timediff_col = 'Predicted_timediff'
        df.rename(columns={'Predicted_timediff': 'timediff'}, inplace=True) # Rename for consistency
    
    if timediff_col:
        df['Step'] = df.groupby('SeqOrder').cumcount()
        df['step_duration'] = df.groupby('SeqOrder')['timediff'].diff().fillna(df['timediff'])
        df['step_duration'] = df['step_duration'].clip(lower=0)
        
        end_marker_step = df[df['sourceID'] == 10].groupby('SeqOrder')['Step'].first()
        df['end_marker_step'] = df['SeqOrder'].map(end_marker_step)
        df.loc[df['Step'] > df['end_marker_step'], 'step_duration'] = 0
        
        df['true_total_time'] = df.groupby('SeqOrder')['step_duration'].transform('sum')

        # Filter out sequences where true_total_time > 1800 seconds
        initial_sequences = df['SeqOrder'].nunique()
        df = df[df['true_total_time'] <= 1800]
        filtered_sequences = df['SeqOrder'].nunique()
        print(f"Filtered out {initial_sequences - filtered_sequences} sequences longer than 1800 seconds.")

        min_timediff = df['timediff'].min()
        max_timediff = df['timediff'].max()
    else:
        print("Warning: Neither 'timediff' nor 'Predicted_timediff' column found in the loaded DataFrame. Skipping true_total_time calculation and filtering.")

    # --- 6. Filter and Reorder Columns ---
    final_columns_to_keep = [
        'SeqOrder', 'sourceID', 'timediff', 'PTAB', 'BodyGroup_from', 'BodyGroup_to',
        'PatientID_from', 'PatientID_to', 'Age', 'Weight', 'Height',
        'Position_encoded', 'Direction_encoded', 'true_total_time' # Add true_total_time
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