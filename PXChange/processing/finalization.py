import pandas as pd

# Load the datasets from your local directory
try:
    predictions_df = pd.read_csv('PXChange/data/176401/predictions_total_time_176401.csv')
    raw_df = pd.read_csv('PXChange/data/176401/176401_raw_new.csv')

    # Columns to add from the raw data file
    columns_to_add = [
        'SN', 'BodyGroup_from', 'BodyGroup_to', 'PatientID_from', 'PatientID_to',
        'PatientId', 'Position', 'Weight', 'Age', 'Height', 'Direction', 'PTAB'
    ]

    # Create a new dataframe based on the predictions file
    merged_df = predictions_df.copy()

    # Replace the 'sourceID' column from predictions_df with the one from raw_df
    # and add the other requested columns.
    # This assumes that the two files are in the same order row-by-row.
    merged_df['sourceID'] = raw_df['sourceID']
    for col in columns_to_add:
        merged_df[col] = raw_df[col]

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv('PXChange/data/176401/FINAL_176401_predictions_with_details.csv', index=False)

    print("Successfully created 'predictions_with_details.csv'")
    print("\nHere's a preview of your new file:")
    print(merged_df.head())

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure that both 'predictions_total_time_176401.csv' and '176401_raw.csv' are in the same directory as this script.")