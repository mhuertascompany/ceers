
import numpy as np
import h5py

import sys
import os

import pandas as pd
import seaborn as sns

data_path = "C:\\Users\\usuario\\Documents\\TFG\\florah_training_SFR\\"
raw_file_path = os.path.join(data_path, "projTNGEAGLESimbamstargt9_random_sizemassSFR_simname.h5")
# New file path for the optimized, cleaned data
processed_file_path = os.path.join(data_path, "florah_eval_SFR_processed_cleaned_data.h5")

def load_and_process_data():
    """ Use this with """
    print("Processing raw data (this happens only once)...")
    
    # We will write directly to the new file to save memory
    with h5py.File(raw_file_path, 'r') as raw_f, \
         h5py.File(processed_file_path, 'w') as proc_f:
        
        for group_name in raw_f:
            group = raw_f[group_name]
            
            # 1. Load raw data into numpy arrays
            # Assumes x_data is shape (N, 3) -> [Mass, Size, SFR]
            x_data = group['x'][:] 
            z_data = group['z'][:]
            sim_name = group['sim'][()]

            # 2. Slice arrays to skip the first row (as per your original code)
            # CRITICAL FIX: We must slice z_data too, otherwise x[1] aligns with z[0]
            x_sliced = x_data[1:]
            z_sliced = z_data[1:]
            
            # 3. Create a Single Boolean Mask (Vectorization)
            # We look for rows that satisfy ALL conditions at once
            # Note: We decode bytes to check for '-' if necessary, or assume direct comparison works
            # depending on h5py string implementation. 
            
            # Extract columns for clarity
            mass_col = x_sliced[:, 0]
            size_col = x_sliced[:, 1]
            sfr_col  = x_sliced[:, 2]

            # Define valid conditions
            # We use numpy logic functions which are much faster than list comprehensions
            is_valid_mass = (mass_col != b'-') & (mass_col != b'-inf')
            is_valid_size = (size_col != b'-') & (size_col != b'-inf') & (size_col.astype(float) > 0)
            is_valid_sfr  = (sfr_col.astype(float) >= 0)

            # Combine conditions (AND logic)
            valid_mask = is_valid_mass & is_valid_size & is_valid_sfr

            # 4. Apply Mask and Transform
            # We apply the mask first, then do the math on the smaller clean subset
            clean_z = z_sliced[valid_mask].astype(float)
            
            clean_mass = mass_col[valid_mask].astype(float)
            
            # Size calculation: value / (1 + z)
            clean_size_raw = size_col[valid_mask].astype(float)
            clean_size = clean_size_raw / (1 + clean_z)
            
            # SFR calculation: value + 1e-5
            clean_sfr = sfr_col[valid_mask].astype(float) + 1e-5
            
            # Time calculation: 1 / (1 + z)
            clean_t = 1 / (1 + clean_z)

            # 5. Stack Features
            # x_copy: [Mass, log10(Size), log10(SFR)]
            x_final = np.column_stack([
                clean_mass, 
                np.log10(clean_size), 
                np.log10(clean_sfr)
            ])
            
            t_final = np.expand_dims(clean_t, 1)

            # 6. Save to the new HDF5 file immediately
            # We use the same group name to preserve structure
            grp_out = proc_f.create_group(group_name)
            grp_out.create_dataset('x', data=x_final)
            grp_out.create_dataset('t', data=t_final)
            # Save simulation name as an attribute or dataset
            grp_out.attrs['sim_name'] = sim_name

    print(f"Data successfully cleaned and saved to: {processed_file_path}")

# --- Main Logic ---

# Check if the processed file already exists
if not os.path.exists(processed_file_path):
    load_and_process_data()
else:
    print("Found processed data file. Loading directly...")