import os
import pandas as pd
import argparse

# Folder containing CSV files
folder_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pcm_configs/"
parser = argparse.ArgumentParser(description='GNOT for operator learning')
parser.add_argument('--csv-dir',type=str,
                    default=folder_path)

args = parser.parse_args()

# Dictionary to store point sets for each file
file_point_sets = {}

folder_path = args.csv_dir

# Iterate through all CSV files in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)

        # Read CSV file (assuming no header)
        df = pd.read_csv(file_path)

        # Extract the last three columns (X, Y, Z coordinates)
        point_cloud = df.iloc[:, -3:].values  # Convert to NumPy array

        # Convert to a set of tuples to ensure uniqueness
        unique_points = set(map(tuple, point_cloud))

        # Store in dictionary
        file_point_sets[file] = unique_points

# Find files with identical point clouds
matching_files = {}
file_list = list(file_point_sets.keys())

for i in range(len(file_list)):
    for j in range(i + 1, len(file_list)):
        file1, file2 = file_list[i], file_list[j]
        
        # Compare point sets
        if file_point_sets[file1] == file_point_sets[file2]:
            if file1 not in matching_files:
                matching_files[file1] = []
            matching_files[file1].append(file2)

# Print matching files
if matching_files:
    print("Files with identical 3D point clouds:")
    for key, matches in matching_files.items():
        print(f"{key} matches with: {matches}")
else:
    print("No files have identical 3D point clouds.")

