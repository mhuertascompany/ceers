import csv
import os
import h5py
import random
import pandas as pd
import numpy as np
import illustris_python as il

basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'
redshifts = np.zeros(99, dtype='float32' )
redshifts.fill(np.nan)
sn=np.zeros(99, dtype='int32' )

for i in range(99):
    h = il.groupcat.loadHeader(basePath,i)
    redshifts[i] = h['Redshift']
    sn[i]=i

# Create a DataFrame to store 'redshifts' and 'sn'
df_snap = pd.DataFrame({'Redshift': redshifts, 'SnapshotNumber': sn})


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
            
            # Get the total number of rows in the file
            rows = list(csv_reader)
            total_rows = len(rows)
            
            # Perform subsampling 15 times
            for subsample_index in range(15):
                # Generate a random starting row index between 0 and 66
                start_row = random.randint(1, min(66, total_rows - 1))
                
                # Generate a random subsample size between 2 and 6 or up to a maximum of 20 rows
                subsample_size = random.randint(2, min(6, total_rows - start_row, 20))
                
                # Extract the rows within the specified range
                selected_rows = rows[start_row:start_row + subsample_size]
                
                # Initialize lists for the second and fifth columns
                second_column = []
                fifth_column = []
                redshifts = []  # List to store the corresponding redshifts
                
                # Extract the values from the second and fifth columns and check for 'inf' values
                for row in selected_rows:
                    if len(row) >= 5 and row[1] != '-inf' and row[4] != '-inf':
                        second_column.append(row[1])  # Index 1 is the second column
                        fifth_column.append(row[4])   # Index 4 is the fifth column
                        
                        # Find the corresponding snapshot number from the row
                        snapshot_number = int(row[1])  # Assuming the snapshot number is in the second column
                        
                        # Find the corresponding redshift from the df_snap DataFrame
                        redshift = df_snap[df_snap['SnapshotNumber'] == snapshot_number]['Redshift'].values[0]
                        redshifts.append(redshift)
            
                # Create a unique index for this subsample entry
                subsample_key = index
                
                # Create a dictionary entry for this subsample
                file_id = filename.split('_')[-1].split('.')[0]  # Extract the identification number
                data_dict[subsample_key] = {'FileID': file_id, 'x': fifth_column, 't': second_column, 'Redshifts': redshifts}
            
                # Increment the index
                index += 1

# Now, data_dict contains the data from all the CSV files with separate entries for each subsample, including unique redshifts

# Save the data_dict to an HDF5 file
hdf5_file_path = directory+'projTNGmstargt9_random.h5'  # Specify the path to your HDF5 file
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    for key, value in data_dict.items():
        group = hdf5_file.create_group(str(key))
        group.create_dataset('FileID', data=value['FileID'])
        group.create_dataset('x', data=value['x'])
        group.create_dataset('t', data=value['t'])
        group.create_dataset('Redshifts', data=value['Redshifts'])

print(f'Data saved to {hdf5_file_path}')



