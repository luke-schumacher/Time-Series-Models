"""
Combines prediction CSVs, undoes encoding, and adds columns from the original data.
"""
import pandas as pd
import os
import glob

def combine_and_denormalize_predictions():
    """
    Combines prediction CSVs, undoes encoding, and adds columns from the original data.
    """
    base_dir = 'C:/Users/lukis/Documents/GitHub/Time-Series-Models/PXChange'
    predictions_dir = base_dir
    data_dir = os.path.join(base_dir, 'data')
    output_file = os.path.join(base_dir, '..\\', 'combined_denormalized_predictions.csv')

    prediction_files = glob.glob(os.path.join(predictions_dir, 'prediction_*_total_time_refactored.csv'))

    sourceid_encoding_legend = {
        'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
        'MRI_FRR_264': 5, 'MRI_FRR_2': 6, 'MRI_FRR_3': 7, 'MRI_FRR_34': 8, 'MRI_MPT_1005': 9,
        'MRI_MSR_100': 10, 'MRI_MSR_104': 11, 'MRI_MSR_21': 12,
        'START': 13,
        'END': 14,
        'MSR_34': 15,
        'FRR_256': 16
    }

    bodygroup_encoding_legend = {
        'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5, 'KNEE': 6,
        'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
    }

    reverse_sourceid_encoding = {v: k for k, v in sourceid_encoding_legend.items()}
    reverse_bodygroup_encoding = {v: k for k, v in bodygroup_encoding_legend.items()}

    all_dfs = []

    for pred_file in prediction_files:
        dataset_id = os.path.basename(pred_file).split('_')[1]
        print(f"Processing dataset: {dataset_id}")

        pred_df = pd.read_csv(pred_file)

        preprocessed_file_path = os.path.join(data_dir, dataset_id, f'preprocessed_{dataset_id}.csv')
        if not os.path.exists(preprocessed_file_path):
            print(f"Warning: Preprocessed file not found for dataset {dataset_id}. Skipping.")
            continue
        preprocessed_df = pd.read_csv(preprocessed_file_path)

        raw_data_dir = os.path.join(data_dir, dataset_id)
        all_csv_files = glob.glob(os.path.join(raw_data_dir, '*.csv'))
        raw_candidates = [f for f in all_csv_files if not any(name in os.path.basename(f).lower() for name in ['encoded', 'filtered', 'predictions', 'preprocessed'])]
        
        if not raw_candidates:
            print(f"Warning: No raw data file found for dataset {dataset_id}. Skipping.")
            continue
            
        raw_data_path = max(raw_candidates, key=os.path.getsize)
        
        try:
            raw_df = pd.read_csv(raw_data_path)
        except Exception:
            raw_df = pd.read_csv(raw_data_path, sep=';')

        # Merge predictions with preprocessed data
        common_cols = list(pred_df.columns.intersection(preprocessed_df.columns))
        merged_df = pd.merge(pred_df, preprocessed_df, on=common_cols, how='left')

        # Get mappings for Position and Direction
        if 'Position' in raw_df.columns:
            pos_labels, pos_uniques = pd.factorize(raw_df['Position'])
            pos_mapping = dict(enumerate(pos_uniques))
            merged_df['Position'] = merged_df['Position_encoded'].map(pos_mapping)

        if 'Direction' in raw_df.columns:
            dir_labels, dir_uniques = pd.factorize(raw_df['Direction'])
            dir_mapping = dict(enumerate(dir_uniques))
            merged_df['Direction'] = merged_df['Direction_encoded'].map(dir_mapping)

        # Undo encoding
        merged_df['sourceID'] = merged_df['sourceID'].map(reverse_sourceid_encoding)
        merged_df['BodyGroup_from'] = merged_df['BodyGroup_from'].map(reverse_bodygroup_encoding)
        merged_df['BodyGroup_to'] = merged_df['BodyGroup_to'].map(reverse_bodygroup_encoding)

        # Add SN column
        merged_df['SN'] = dataset_id

        # Select and rename columns
        final_columns = [
            'SN', 'SeqOrder', 'Step', 'sourceID', 'timediff', 'predicted_total_time', 'true_total_time',
            'BodyGroup_from', 'BodyGroup_to', 'PatientID_from', 'PatientID_to',
            'Age', 'Weight', 'Height', 'Position', 'Direction'
        ]
        
        for col in final_columns:
            if col not in merged_df.columns:
                merged_df[col] = None
        
        merged_df = merged_df[final_columns]

        all_dfs.append(merged_df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Successfully combined and denormalized predictions to {output_file}")
    else:
        print("No prediction files were processed.")

if __name__ == "__main__":
    combine_and_denormalize_predictions()
