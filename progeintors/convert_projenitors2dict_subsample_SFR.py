import csv
import os
import h5py
import random
import pandas as pd
import numpy as np
import illustris_python as il
import pdb

# redshift - snapshot for TNG
basePath = '/virgotng/universe/IllustrisTNG/TNG100-1/output/'
redshifts = np.zeros(99, dtype='float32' )
redshifts.fill(np.nan)
sn=np.zeros(99, dtype='int32' )

for i in range(1,100):
    h = il.groupcat.loadHeader(basePath,i)
    redshifts[i-1] = h['Redshift']
    sn[i-1]=i

# Create a DataFrame to store 'redshifts' and 'sn'
print(sn)
print(redshifts)
df_snap = pd.DataFrame({'Redshift': redshifts, 'SnapshotNumber': sn})


# redshift - snapshot for EAGLE
basePath = '/virgotng/universe/Eagle/Eagle100-1/output'
redshifts = np.zeros(28, dtype='float32' )
redshifts.fill(np.nan)
sn=np.zeros(28, dtype='int32' )

for i in range(1,29):
    h = il.groupcat.loadHeader(basePath,i)
    redshifts[i-1] = h['Redshift']
    sn[i-1]=i

# Create a DataFrame to store 'redshifts' and 'sn'
print(sn)
print(redshifts)
df_snap_eagle = pd.DataFrame({'Redshift': redshifts, 'SnapshotNumber': sn})

# redshift - snapshot for SIMBA
basePath = '/virgotng/universe/Simba/L100n1024FP/output'
redshifts = np.zeros(151, dtype='float32' )
redshifts.fill(np.nan)
sn=np.zeros(151, dtype='int32' )

for i in range(1,152):
    h = il.groupcat.loadHeader(basePath,i)
    redshifts[i-1] = h['Redshift']
    sn[i-1]=i

# Create a DataFrame to store 'redshifts' and 'sn'
print(sn)
print(redshifts)
df_snap_simba = pd.DataFrame({'Redshift': redshifts, 'SnapshotNumber': sn})



df_list=[df_snap_simba,df_snap_eagle,df_snap,df_snap]
sim_list = ["SIMBA","EAGLE","TNG","TNG"]
max_snap = [78,6,33,33]

# Define a dictionary to store the data
data_dict = {}

output_path = '/u/mhuertas/data/CEERS/'

# Directory containing the CSV files
directory_list = ['/u/mhuertas/data/CEERS/Simbaprojenitors_sizemass_sSFR','/u/mhuertas/data/CEERS/EAGLEprojenitors_sizemass_sSFR','/u/mhuertas/data/CEERS/TNG100projenitors_sizemass_sSFR','/u/mhuertas/data/CEERS/TNG50projenitors_sizemass_sSFR']

# Initialize an index for arbitrary numbering
index = 0

for directory,df,simul,m in zip(directory_list,df_list,sim_list,max_snap):
    print('Doing folder '+ directory)
# Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith(simul):
            file_path = os.path.join(directory, filename)
            
            # Open and read the CSV file
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                
                # Get the total number of rows in the file
                rows = list(csv_reader)
                total_rows = len(rows)
                
                # Perform subsampling 15 times
                for subsample_index in range(15):
                    # Generate a random starting row index between 1 and 33
                    start_row = random.randint(1, min(m, total_rows - 1))
                    
                    # Generate a random step size (x) between 1 and 3
                    step_size = random.randint(1, 3)
                    if m<30:
                        step_size=1
                    
                    
                    # Initialize lists for the second and fifth columns
                    second_column = []
                    fifth_column = []
                    sixth_column = []
                    seventh_column=[]
                    redshifts = []  # List to store the corresponding redshifts
                    scale_factor=[]
                    
                    # Extract the values from the second and fifth columns and check for 'inf' values
                    for i in range(start_row, min(start_row + 20*step_size, total_rows), step_size):
                        row = rows[i]
                        if len(row) >= 5 and row[1] != '-inf' and row[4] != '-inf':
                            second_column.append(row[1])  # Index 1 is the second column
                            fifth_column.append(row[6])   # Index 6 is the seventh column (mass)
                            sixth_column.append(row[5])   # Index 5 is the fifth column (size)
                            seventh_column.append(row[7])   # Index 7 is the eight column (SFR)
                            
                            # Find the corresponding snapshot number from the row
                            snapshot_number = int(row[1])  # Assuming the snapshot number is in the second column
                            
                            # Find the corresponding redshift from the df_snap DataFrame
                            #print(snapshot_number)
                            redshift = df[df['SnapshotNumber'] == snapshot_number]['Redshift'].values[0]
                            redshifts.append(redshift)
                            scale_factor.append(1/(1+redshift))
                
                    # Create a unique index for this subsample entry
                    subsample_key = index
                    
                    # Create a dictionary entry for this subsample
                    file_id = filename.split('_')[-1].split('.')[0]  # Extract the identification number
                    if len(redshifts)>5:
                        x=np.zeros((len(fifth_column),3))
                        #print(x.shape)
                        x[:,0]=fifth_column
                        x[:,1]=sixth_column
                        x[:,2]=seventh_column

                        data_dict[subsample_key] = {'FileID': file_id, 'x': x, 'snapshot': second_column, 'z': redshifts, 't': scale_factor}
                
                    # Increment the index
                    index += 1
                print('Done '+file_id)

# Now, data_dict contains the data from all the CSV files with separate entries for each subsample, including unique redshifts

# Save the data_dict to an HDF5 file
hdf5_file_path = output_path+'projTNGEAGLESimbamstargt9_random_sizemassSFR.h5'  # Specify the path to your HDF5 file
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    for key, value in data_dict.items():
        group = hdf5_file.create_group(str(key))
        group.create_dataset('FileID', data=value['FileID'])
        group.create_dataset('x', data=value['x'])
        group.create_dataset('snapshot', data=value['snapshot'])
        group.create_dataset('z', data=value['z'])
        group.create_dataset('t', data=value['t'])


print(f'Data saved to {hdf5_file_path}')



