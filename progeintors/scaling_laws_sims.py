# to be run in max planck servers

import illustris_python as il
import pandas as pd
from astropy.io import fits
import numpy as np
import pdb
import h5py

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

df_list=[df_snap,df_snap_eagle,df_snap_simba]
sim_list = ["TNG","EAGLE","Simba"]
basePath_list=['/virgotng/universe/IllustrisTNG/TNG100-1/output/','/virgotng/universe/Eagle/Eagle100-1/output/','/virgotng/universe/Simba/L100n1024FP/output/']


redshifts = [0.5,1,2,3,4,5]

# Define the name of the single HDF5 file
output_dir = "/u/mhuertas/data/CEERS"
hdf5_file_path = output_dir+"/all_simulations_data.hdf5"

# Assuming basePath_list, sim_list, df_list, and redshifts are defined
for z in redshifts:
    for basePath, sim, df_snap_simba in zip(basePath_list, sim_list, df_list):
        # Calculate the absolute difference between the given redshift and each redshift in the DataFrame
        df_snap_simba['abs_diff'] = (df_snap_simba['Redshift'] - z).abs()

        # Find the index of the smallest difference
        closest_index = df_snap_simba['abs_diff'].idxmin()

        # Use this index to find the corresponding snapshot number
        closest_snapshot = df_snap_simba.loc[closest_index, 'SnapshotNumber']
        fields = ['SubhaloMassInRadType', 'SubhaloHalfmassRadType', 'SubhaloSFRinRad']
        subhalos = il.groupcat.loadSubhalos(basePath, closest_snapshot, fields=fields)

        # Extract and process the subhalo data
        mass = np.log10(subhalos['SubhaloMassInRadType'][:, 4] * 1e10 / 0.704)
        exact_redshift = df_snap_simba[df_snap_simba['SnapshotNumber'] == closest_snapshot]['Redshift'].values[0]
        re = np.log10(subhalos['SubhaloHalfmassRadType'][:, 4] * 0.704 / (1 + exact_redshift))
        sfr = np.log10(subhalos['SubhaloSFRinRad'])

        # Create a DataFrame
        df_subhalos = pd.DataFrame({'Mass': mass, 'Re': re, 'SFR': sfr})

        # Open the single HDF5 file in 'append' mode
        with h5py.File(hdf5_file_path, 'a') as hdf5_file:
            # Convert DataFrame to a record array
            record_array = df_subhalos.to_records(index=False)
            group_name = f"{sim}/z{z:.2f}"  # Group name, e.g., "Illustris1/z0.50"
            
            # Create a group and dataset for the DataFrame in the file
            if group_name not in hdf5_file:  # Check if the group already exists
                group = hdf5_file.create_group(group_name)
            else:
                group = hdf5_file[group_name]
            
            # Create a dataset for the DataFrame in the group
            dataset_name = 'Subhalos'
            if dataset_name not in group:  # Check if the dataset already exists
                group.create_dataset(dataset_name, data=record_array, compression="gzip")
            else:
                # If dataset exists, you can decide to overwrite or update it
                del group[dataset_name]  # Remove the existing dataset
                group.create_dataset(dataset_name, data=record_array, compression="gzip")


