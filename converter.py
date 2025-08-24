import pandas as pd

# Specify the file paths
data_file = "C:/Users/capta/Downloads/Model/wdbc.data"  # Replace with your .data file name
csv_file = "C:/Users/capta/Downloads/Model/wdbc.csv"    # Replace with desired .csv file name

# Load the .data file into a pandas DataFrame
# Adjust the delimiter parameter if needed (e.g., ',' for comma, '\t' for tab, or ' ' for space)
df = pd.read_csv(data_file, delimiter=',', header=None)

# Save the DataFrame to a .csv file
df.to_csv(csv_file, index=False)

print(f"Converted {data_file} to {csv_file} successfully!")
