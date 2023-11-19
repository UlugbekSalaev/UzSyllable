import pandas as pd

# Replace 'input_file.csv' with the path to your input CSV file
input_file_path = 'dataset_latin.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file_path)

# Select the first and second columns
selected_columns = df.iloc[:, :2]

# Replace 'output_file.csv' with the desired path for the output CSV file
output_file_path = 'dataset_latinCV.csv'

# Save the selected columns to another CSV file
selected_columns.to_csv(output_file_path, index=False)

print(f"Selected columns saved to {output_file_path}")
