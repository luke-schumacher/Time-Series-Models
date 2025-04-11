# combine_csvs_in_subdir.py
import pandas as pd
import glob
import os
import sys # To get script name and exit

# --- Configuration ---
# Path to the folder containing the CSV files, relative to where you run the script.
# IMPORTANT: Run this script from the directory ABOVE 'PXChange'.
FOLDER_PATH = 'PXChange/model_results'

# Base name for the combined output file
OUTPUT_BASE_NAME = 'combined_model_results'
# Maximum number of SNs to list in the filename directly
MAX_SNS_IN_FILENAME = 5
# --- End Configuration ---

print("--- Starting CSV Combination Script ---")

# Check if the target folder exists before proceeding
if not os.path.isdir(FOLDER_PATH):
    print(f"\nError: The specified folder does not exist: '{FOLDER_PATH}'")
    print(f"Please ensure you are running this script from the directory *containing* the '{os.path.dirname(FOLDER_PATH)}' folder.")
    print(f"Current working directory is: '{os.getcwd()}'")
    sys.exit()

print(f"Script will look for CSVs inside: '{os.path.abspath(FOLDER_PATH)}'")

# Get the name of this script file to avoid accidentally processing it if it's somehow in the target folder
try:
    script_name = os.path.basename(__file__)
except NameError:
    script_name = 'combine_csvs_in_subdir.py' # Adjust if you rename the script
    print(f"Warning: Assuming script name is '{script_name}'.")

# Find all files ending with .csv IN THE SPECIFIED FOLDER_PATH
search_pattern = os.path.join(FOLDER_PATH, '*.csv')
all_csv_files = glob.glob(search_pattern)

# --- Filter out previous combined outputs ---
# (No need to filter script name here unless you place the script inside FOLDER_PATH, which is not the instruction now)
print("\nFiltering file list (removing potential previous outputs)...")
filtered_csv_files = []
potential_output_prefix = os.path.join(FOLDER_PATH, OUTPUT_BASE_NAME.lower() + "_sn_") # Check for prefix within the target folder

for f_path in all_csv_files:
    basename = os.path.basename(f_path) # Get just the filename for checking prefix
    # Exclude files that look like previous outputs of this script
    if basename.lower().startswith(OUTPUT_BASE_NAME.lower() + "_sn_"):
         print(f" - Ignoring potential previous output: {basename}")
         continue
    # Keep the file path for processing
    filtered_csv_files.append(f_path)


if not filtered_csv_files:
    print(f"\nError: No CSV files found to combine inside '{FOLDER_PATH}'.")
    print("Please ensure:")
    print(f" 1. You are running the script from the directory containing '{os.path.dirname(FOLDER_PATH)}'.")
    print(f" 2. Your individual model result CSV files are located inside '{FOLDER_PATH}'.")
    print(" 3. The CSV files have the '.csv' extension.")
    sys.exit() # Exit if no relevant files found

print(f"\nFound {len(filtered_csv_files)} CSV files to combine in '{FOLDER_PATH}':")
for f in filtered_csv_files:
    print(f" - {os.path.basename(f)}") # Print just the filename, not the full path

# --- Read each valid CSV file into a list of DataFrames ---
list_of_dataframes = []
serial_numbers = set() # Use a set to store unique SNs found

print("\nReading individual CSV files...")
# The file paths in filtered_csv_files already include FOLDER_PATH from glob
for file_path in filtered_csv_files:
    filename = os.path.basename(file_path) # For display messages
    try:
        # Read the current CSV file using its full path
        df = pd.read_csv(file_path)

        if not df.empty:
            print(f"  - Reading {filename} ({len(df)} rows, {len(df.columns)} columns)")
            list_of_dataframes.append(df)
            if 'SN' in df.columns:
                valid_sns = df['SN'].dropna().unique()
                if len(valid_sns) > 0:
                    serial_numbers.update(map(str, valid_sns))
                else:
                     print(f"    'SN' column found but contains no values in {filename}.")
            else:
                print(f"    Warning: 'SN' column not found in {filename}.")
        else:
            print(f"  - Skipping empty file: {filename}")

    except pd.errors.EmptyDataError:
        print(f"  - Skipping empty or invalid file: {filename}")
    except Exception as e:
        print(f"  - Error reading {filename}: {e}. Skipping.")

# --- Combine (concatenate) the DataFrames ---
if not list_of_dataframes:
    print("\nError: No data could be read from any CSV files. Cannot create combined file.")
    sys.exit()

print("\nCombining data from all read files...")
try:
    combined_df = pd.concat(list_of_dataframes, ignore_index=True)
    print(f"-> Successfully combined data.")
    print(f"-> The final combined table has {len(combined_df)} rows and {len(combined_df.columns)} columns.")
except Exception as e:
    print(f"Error during data combination: {e}")
    sys.exit()

# --- Determine the output filename including SNs ---
sorted_sns = sorted(list(serial_numbers))
output_filename_base = "" # Just the filename part, without the folder path

if sorted_sns:
    sn_part = '_'.join(sorted_sns[:MAX_SNS_IN_FILENAME])
    if len(sorted_sns) > MAX_SNS_IN_FILENAME:
        sn_part += '_etc'
    output_filename_base = f"{OUTPUT_BASE_NAME}_SN_{sn_part}.csv"
    print(f"\nGenerated output filename based on found SNs: '{output_filename_base}'")
else:
    output_filename_base = f"{OUTPUT_BASE_NAME}_combined_NoSNFound.csv"
    print(f"\nWarning: No SNs found in any 'SN' column. Using generic filename: '{output_filename_base}'")

# --- Save the combined DataFrame to a CSV file INSIDE FOLDER_PATH ---
# Construct the full path for the output file
full_output_path = os.path.join(FOLDER_PATH, output_filename_base)
absolute_output_path = os.path.abspath(full_output_path) # Get absolute path for clear message

print(f"\nSaving combined data to: {absolute_output_path}") # Show the full path

try:
    # Save the file to the calculated full path
    combined_df.to_csv(full_output_path, index=False)
    print(f"\n--- Success! Combined CSV file saved successfully. ---")
except Exception as e:
    print(f"\n--- Error! Failed to save the combined file: {e} ---")
    # Provide hint if it's a directory issue
    if isinstance(e, FileNotFoundError):
         print(f"Hint: Ensure the directory '{os.path.dirname(full_output_path)}' exists and you have write permissions.")