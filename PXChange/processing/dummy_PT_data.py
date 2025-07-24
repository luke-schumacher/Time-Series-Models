import pandas as pd
import numpy as np
# No need for 'io' if reading directly from a file path

# Define the input file name.
# Ensure this CSV file is in the same directory as your Python script,
# or provide the full path to the file.
input_filename = 'PXChange/data/175651/encoded_175651_condensed.csv'

try:
    # Read the CSV file directly from the specified path.
    df = pd.read_csv(input_filename)

    # Get the number of rows in the DataFrame.
    num_rows = len(df)

    # Generate dummy data for the new 'patient_height' column.
    # Heights are randomly generated integers between 150 cm and 190 cm (inclusive).
    df['patient_height'] = np.random.randint(150, 191, size=num_rows)

    # Generate dummy data for the new 'patient_weight' column.
    # Weights are randomly generated integers between 50 kg and 100 kg (inclusive).
    df['patient_weight'] = np.random.randint(50, 101, size=num_rows)

    # Generate dummy data for the new 'patient_age' column.
    # Ages are randomly generated integers between 18 years and 90 years (inclusive).
    df['patient_age'] = np.random.randint(18, 91, size=num_rows)

    # Generate dummy data for the new 'patient_gender' column.
    # Genders are represented as 0 or 1, randomly assigned.
    df['patient_gender'] = np.random.randint(0, 2, size=num_rows)

    # Define the name for the output CSV file.
    output_filename = 'PXChange/data/175651/encoded_175651_condensed_with_dummy_data.csv'
    

    # Save the modified DataFrame to a new CSV file.
    # index=False prevents pandas from writing the DataFrame index as a column in the CSV.
    df.to_csv(output_filename, index=False)

    # Print a confirmation message and the first few rows of the updated DataFrame.
    print(f"Successfully added dummy data and saved to {output_filename}")
    print("First 5 rows of the new dataset:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please make sure the CSV file is in the same directory as this Python script, or provide the full path to the file.")
except Exception as e:
    # Catch any other potential errors during processing and print an error message.
    print(f"An unexpected error occurred: {e}")