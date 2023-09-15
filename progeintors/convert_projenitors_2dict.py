import csv
import os
import h5py

# Define a dictionary to store the data
data_dict = {}

# Directory containing the CSV files
directory = '/u/mhuertas/data/CEERS/TNG100projenitors'

# Initialize an index for arbitrary numbering
index = 0

# Loop through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv") and filename.startswith("TNG100_tree_"):
        file_path = os.path.join(directory, filename)
        
        # Open and read the CSV file
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Initialize lists for the second and fifth columns
            second_column = []
            fifth_column = []
            
            # Read each row in the CSV file
            for row in csv_reader:
                # Check if the row has at least 5 columns
                if len(row) >= 5:
                    # Check for 'inf' values in the columns
                    if row[1] != 'inf' and row[4] != 'inf':
                        # Append the values from the second and fifth columns to the lists
                        second_column.append(row[1])  # Index 1 is the second column
                        fifth_column.append(row[4])   # Index 4 is the fifth column
            
            # Create a dictionary entry for this file
            file_id = filename.split('_')[-1].split('.')[0]  # Extract the identification number
            data_dict[index] = {'FileID': file_id, 'x': fifth_column, 't': second_column}
            
            # Increment the index
            index += 1


# Save the data_dict to an HDF5 file
hdf5_file_path = 'data.h5'  # Specify the path to your HDF5 file
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    for key, value in data_dict.items():
        group = hdf5_file.create_group(str(key))
        group.create_dataset('FileID', data=value['FileID'])
        group.create_dataset('x', data=value['x'])
        group.create_dataset('t', data=value['t'])

print(f'Data saved to {hdf5_file_path}')

