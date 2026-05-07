import numpy as np

import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm

import torch
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
import h5py

import sys
import os
sys.path.append(r"C:\Users\usuario\Documents\TFG\florah\src")

import florah
from florah import utils
from florah.models import rnn_model
from florah.models.rnn_model import rnn_generator
from florah.models.rnn_model.rnn_generator import DataModule 


from astropy.cosmology import Planck13
import astropy.units as u

import random
import pandas as pd
import seaborn as sns


# Load the trained model from a checkpoint file
checkpoint_path = "/scratch/lmarrero-ext/CEERS_train/proj/TNGEagleSimba_mass_size_gt9/SFR_val/last-v1.ckpt"  # Specify the path to your checkpoint file
loaded_model = DataModule.load_from_checkpoint(checkpoint_path,map_location='cpu', weights_only=False)
# Set the model to evaluation mode (important if you have dropout or batch normalization layers)
loaded_model.eval()


data_path = "/scratch/lmarrero-ext/florah_eval/"
# New file path for the optimized, cleaned data
processed_file_path = os.path.join(data_path, "florah_eval_SFR_processed_cleaned_data.h5")

# Load the data from the clean file into your lists
x = []
t = []
file = []

with h5py.File(processed_file_path, 'r') as hdf5_file:
    
    for group_name in hdf5_file:
        group = hdf5_file[group_name]
        
        x.append(group['x'][:])
        t.append(group['t'][:])
        
        # Decode the byte string back to regular string
        file.append(str(group.attrs['sim_name']))

print(f"Loaded {len(x)} groups ready for training.")



# Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
node_features = {'x': [np.array(arr, dtype=np.float32) for arr in x], 't': [np.array(arr, dtype=np.float32) for arr in t]}

# Now, 'x' and 't' contain cleaned and converted data as NumPy arrays of objects
x = node_features['x']   # stellar mass, half mass radius and SFR
t = node_features['t']   # scale factor

# Split the data into training (85%) and validation (15%) sets
x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.15, random_state=42)

x_train, x_val, t_train, t_val, file_train, file_val = train_test_split(x, t, file, test_size=0.15, random_state=42)


# Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
node_features_train = {'x': [np.array(arr, dtype=np.float32) for arr in x_train], 't': [np.array(arr, dtype=np.float32) for arr in t_train]}
node_features_val = {'x': [np.array(arr, dtype=np.float32) for arr in x_val], 't': [np.array(arr, dtype=np.float32) for arr in t_val]}


num_elements = 10000
start_index = 3

# Defining empty lists to do storing:

# master_z_list will be a 1D array
# master_diff_list will be an array with dims (N, 3)
# master_diff_median_list will be an array with dims (N, 3)

master_z_list = []
master_diff_list = []
master_diff_median_list = []

for i, pos in enumerate(range(start_index, start_index + num_elements)):
    # --- SAFETY CHECK: Skip empty galaxies ---
    if len(x_val[pos]) == 0:
        print(f"Skipping index {pos}: galaxy has no data (len=0).")
        continue 
    
    # --- A. Prepare data and sample trees ---
    n_trees = 50
    root = x_val[pos][0]
    roots = np.tile(root, (n_trees, 1))

    # We compute z_init to set up the model grid
    z_init_val = (1 / t_val[pos] - 1)[0].item()
    z_init_val = min(z_init_val, 6)
    
    # Redshift grid for model
    redshifts_model = np.arange(z_init_val, 6.5, 0.2) 

    scale_factors = 1 / (1 + redshifts_model)
    scale_factors = np.repeat(scale_factors[None, :], n_trees, axis=0)
    scale_factors = scale_factors[..., np.newaxis]

    sampled_trees = utils.sampling.sample_trees(loaded_model, roots, scale_factors)
    means_trees = np.mean(sampled_trees, axis=0) # Mean model prediction on the grid.
    median_trees = np.median(sampled_trees, axis=0)

    # --- B. Interpolate to simulation's grid ---
    z_val = 1 / t_val[pos] - 1 
    z_val = z_val.flatten()

    mean_interpolated = np.zeros_like(x_val[pos])
    median_interpolated = np.zeros_like(x_val[pos])

    for col in range(3):
        # We interpolate each physical property separately
        mean_interpolated[:, col] = np.interp(z_val, redshifts_model, means_trees[:, col])
        median_interpolated[:, col] = np.interp(z_val, redshifts_model, median_trees[:, col])

    # --- C. Calculate differences ---
    difference = x_val[pos] - mean_interpolated
    difference_median = x_val[pos] - median_interpolated

    # --- D. Storing ---
    # We do not append the entire matrix; instead, we extend the list.
    # This is important because otherwise we cannot compute the mean of the columns 
    # (if each galaxy has) different points to plot.
    master_z_list.extend(z_val)
    master_diff_list.extend(difference)
    master_diff_median_list.extend(difference_median)


# We convert to np.array in order to compute means later
all_z_numpy = np.array(master_z_list)                        # Shape: (N,)
all_diff_numpy = np.array(master_diff_list)                  # Shape: (N, 3)
all_diff_median_numpy = np.array(master_diff_median_list)    # Shape: (N, 3)



# --- Change storing directory ---
# Specifying in which path I want the plots and diff data to be saved.
run_directory = 'num_elements='+str(num_elements)+' n_trees='+str(n_trees)

#Combine the paths safely
full_dir_path = os.path.join(data_path, run_directory)

# create the directory if it doesn't exist yet
os.makedirs(full_dir_path, exist_ok=True)


# Open a new HDF5 file ( 'w' for write)
with h5py.File(full_dir_path+'/differences.h5py', 'w') as f:
    # Create datasets
    f.create_dataset('z_values', data=all_z_numpy, compression="gzip", compression_opts=5)
    f.create_dataset('diffs', data=all_diff_numpy, compression="gzip", compression_opts=5)
    f.create_dataset('median_diffs', data=all_diff_median_numpy, compression="gzip", compression_opts=5)
    
    # You can even add metadata (attributes)
    f.attrs['description'] = 'Model output comparison data - Differences'
    f.attrs['date'] = '2026-05-07'