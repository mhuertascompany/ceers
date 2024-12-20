import numpy as np
import torch
import pytorch_lightning as pl

import florah
from florah import utils
from florah.models import rnn_model
from florah.models.rnn_model import rnn_generator

import h5py

torch.set_default_dtype(torch.float32)  
from sklearn.model_selection import train_test_split


## READ THE TRAINING DATA


print('Loading training data..')
# Initialize an empty dictionary to store the loaded data
loaded_data_dict = {}
data_path = "/scratch/mhuertas/CEERS/proj/"

hdf5_file_path = data_path+"projTNGEAGLEmstargt9_random_sizemass.h5"
# Initialize 'x' and 't' as lists to store the cleaned data
x = []
t = []
node_features = {'x': None, 't': None}
# Open the HDF5 file for reading
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Loop through the groups (indexed by integers)
    for group_name in hdf5_file:
        group = hdf5_file[group_name]
        
        # Read the 'x' and 't' data from the group
        x_data = group['x'][:]
        t_data = 1/(1+group['z'][:])
        z_data = group['z'][:]
        
        # Convert the 'x_data' to a list of floats while ignoring non-numeric and 'inf' values and skipping the first row
        cleaned_x_mass = [float(value) for value,size in zip(x_data[1:,0],x_data[1:,1]) if value != b'-' and value != b'-inf' and size>0]
        cleaned_x_size = [float(value/(1+z)) for value,z in zip(x_data[1:,1],z_data) if value != b'-' and value != b'-inf' and value >0]
        
        #print(np.array(cleaned_x_mass).shape)
        x_copy = np.column_stack([cleaned_x_mass, np.log10(cleaned_x_size)])
        #print(np.array(x_copy).shape)
        #cleaned_x = [float(value) for value in x_data[1:] if value != b'-' and value != b'-inf']
        # Convert the 't_data' to a list of floats while ignoring non-numeric and 'inf' values and skipping the first row
        cleaned_t = [float(value) for value,size in zip(t_data[1:],x_data[1:,1]) if value != b'-' and value != b'-inf' and size>0]
        cleaned_t = np.expand_dims(cleaned_t,1)
        
       # Append the cleaned 'x' and 't' data to their respective lists
        x.append(x_copy)
        t.append(cleaned_t)





# Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
node_features = {'x': [np.array(arr, dtype=np.float32) for arr in x], 't': [np.array(arr, dtype=np.float32) for arr in t]}

# Now, 'x' and 't' contain cleaned and converted data as NumPy arrays of objects


x = node_features['x']   # stellar mass and half mass radius
t = node_features['t']   # scale factor

# Split the data into training (85%) and validation (15%) sets
x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.15, random_state=42)


# Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
node_features_train = {'x': [np.array(arr, dtype=np.float32) for arr in x_train], 't': [np.array(arr, dtype=np.float32) for arr in t_train]}
node_features_val = {'x': [np.array(arr, dtype=np.float32) for arr in x_val], 't': [np.array(arr, dtype=np.float32) for arr in t_val]}


print('Training...')

#### TRAININIG

# define hyperparameters
# model architecture
model_hparams = dict(
    in_channels=2,   # number of input channels, in this case it is the halo mass and concentration
    out_channels=2,   # number of output channels, in this case it is also the halo mass and concentration
    num_layers=4,
    hidden_features=128,
    num_layers_flows=4,
    hidden_features_flows=128,
    num_blocks=2,
    rnn_name="GRU",
)
# optimizer
optimizer_hparams = dict(
    optimizer=dict(
        optimizer="AdamW",
        lr=5e-4,
        betas=(0.9, 0.98)
    ),
    scheduler=dict(
        scheduler="ReduceLROnPlateau",
        patience=10
    )
)
# time series preprocessing transformation
transform_hparams = dict(
    nx=2, 
    ny=2,
    sub_dim=[0,1]
)

# Now we can create the model. The model is a Pytorch Lightning module, which 
# will store the hyperparameters and the optimizer.
model = rnn_generator.DataModule(
    model_hparams, transform_hparams, optimizer_hparams
)


# preprocess the data and create DataLoader
# new node features: x, y, t, seq_len, mask
# x: stellar mass and size at the current time step (normalized)
# y: accreted mass and size at next time step (normalized)
# t: normalized time
# seq_len: length of each time series
# mask: mask for padding
preprocessed_node_features = model.transform(node_features_train, fit=True)
preprocessed_node_features_val = model.transform(node_features_val, fit=True)

dataset = torch.utils.data.TensorDataset(*preprocessed_node_features)
dataset_val = torch.utils.data.TensorDataset(*preprocessed_node_features_val)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1024, shuffle=True, num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1024, shuffle=False, num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

# Create a Pytorch lightning trainer. This will handle the training loop and
# checkpointing.
trainer  = pl.Trainer(
    default_root_dir="/scratch/mhuertas/CEERS/proj/TNGEagle_mass_size_gt9",
    accelerator="auto",
    devices=1,
    max_epochs=500,
    logger=pl.loggers.CSVLogger("delta_run", name="delta_run"),
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath='/scratch/mhuertas/CEERS/proj/TNGEagle_mass_size_gt9/delta_val/',filename="{epoch}-{val_loss:.4f}", save_weights_only=False,
            mode="min", monitor="val_loss",save_top_k=5,save_last=True,every_n_epochs = 1),
        pl.callbacks.LearningRateMonitor("epoch"),
    ],
    enable_progress_bar=True,
)



# Start training
trainer.fit(
    model=model, train_dataloaders=data_loader,
    val_dataloaders=data_loader_val)


