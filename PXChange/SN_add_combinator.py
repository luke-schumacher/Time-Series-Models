import pandas as pd
import os
import re # Import regular expressions module

def combine_csvs_with_sn():
    """
    Combines three specified CSV files into one, adding a unique serial number 
    column to each before merging. File paths and SNs are hardcoded.
    The serial number is extracted from the filename.
    """
    print("--- CSV Combining Script with Hardcoded Files ---")

    # Hardcoded file information
    # Each tuple: (filename, serial_number)
    # The serial number will be extracted from the filename.
    files_to_process = [
        "PxChange/model_results_SN_175974.csv",
        "PxChange/model_results_SN_185625.csv",
        "PxChange/model_results_SN_175651.csv"
    ]

    all_dataframes = []

    # Process each file
    for file_path in files_to_process:
        # Extract SN from filename using regex (looks for 6 digits after "_SN_")
        match = re.search(r'_SN_(\d{6})\.csv$', file_path)
        if not match:
            print(f"Error: Could not extract a 6-digit serial number from filename '{file_path}'. Skipping this file.")
            print(f"Filename must be in the format '..._SN_XXXXXX.csv' where XXXXXX is the 6-digit SN.")
            continue # Skip to the next file

        sn = match.group(1) # The extracted 6-digit serial number

        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'. Please ensure the file exists in the same directory as the script, or provide the full path if it's elsewhere.")
            print("If the files are in a different directory, you might need to adjust the file_path strings in the 'files_to_process' list to include the full path.")
            continue # Skip to the next file

        try:
            print(f"\nProcessing '{os.path.basename(file_path)}' with SN: {sn}...")
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"Warning: File '{file_path}' is empty. It will be included, but will contribute no data.")
                # Add SN column even if empty, to maintain structure if other files have data
                df['SN'] = sn 
                 # Ensure 'SN' is the first column
                if 'SN' in df.columns: # Check if SN column was added
                    cols = ['SN'] + [col for col in df.columns if col != 'SN']
                    df = df[cols]
            else:
                # Add the serial number column
                # Ensure 'SN' column is the first column
                df.insert(0, 'SN', sn)

            all_dataframes.append(df)
            print(f"Successfully processed and added SN to '{os.path.basename(file_path)}'.")
            if not df.empty:
                print(f"First few rows of '{os.path.basename(file_path)}' with SN:")
                print(df.head(2))
            else:
                print(f"'{os.path.basename(file_path)}' was empty.")
            print("-" * 30)

        except pd.errors.EmptyDataError: # Should be caught by the df.empty check now
            print(f"Warning: File '{file_path}' is empty and could not be processed conventionally. SN column added.")
            # Create an empty DataFrame with just the SN column if read_csv fails on empty file
            df_empty_sn = pd.DataFrame({'SN': [sn]})
            all_dataframes.append(df_empty_sn)
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")
            # Optionally, decide if one error should stop the whole process
            # return

    if not all_dataframes:
        print("No dataframes were successfully processed. Exiting.")
        return

    # Combine all dataframes
    try:
        print("\nCombining all processed files...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("Files combined successfully.")
        
        if combined_df.empty:
            print("The combined DataFrame is empty (all source files might have been empty or failed processing).")
        else:
            print("\nFirst few rows of the combined data:")
            print(combined_df.head())
            print("\nLast few rows of the combined data:")
            print(combined_df.tail())
        print(f"\nTotal rows in combined file: {len(combined_df)}")


    except Exception as e:
        print(f"An error occurred while combining the DataFrames: {e}")
        return

    # Get output file name
    while True:
        output_filename = input("\nEnter the desired name for the combined CSV file (e.g., 'combined_data.csv'): ")
        if output_filename.lower().endswith('.csv') and output_filename.strip(): # Also check if not just whitespace
            break
        else:
            print("Please ensure the filename ends with '.csv' and is not empty.")

    # Save the combined dataframe to a new CSV
    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"\nCombined data successfully saved to '{output_filename}'")
        print(f"Full path: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"An error occurred while saving the combined file: {e}")

if __name__ == "__main__":
    # This ensures the script runs when executed directly
    combine_csvs_with_sn()
