import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import pdb
import matplotlib.pyplot as plt
import h5py    
import pandas as pd

#import sklearn


import seaborn as sns
print(sns.__version__)
#import plotly.express as px
from scipy.special import betaincinv
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


import astropy.units as u
from astropy.cosmology import Planck15  

from florah.models.rnn_model.rnn_generator import DataModule 

data_path = "/scratch/mhuertas/CEERS/data_release/cats/"
ceers_cat = pd.read_csv(data_path+"CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v051_bug.csv")







morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_f356w_mean,ceers_cat.disk_f356w_mean,ceers_cat.irr_f356w_mean,ceers_cat.bd_f356w_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
    
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f356>0.3)]=1
#morph_flag[(ceers_cat.irr_f356>0.3) & (ceers_cat.sph_f356>0.3)]=3
ceers_cat['morph_flag_f356w']=np.array(morph_flag)
ceers_cat['delta_f356']=np.array(delta_value)

morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_f200w_mean,ceers_cat.disk_f200w_mean,ceers_cat.irr_f200w_mean,ceers_cat.bd_f200w_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f200w']=np.array(morph_flag)
ceers_cat['delta_f200']=np.array(delta_value)

morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_f444w_mean,ceers_cat.disk_f444w_mean,ceers_cat.irr_f444w_mean,ceers_cat.bd_f444w_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f444w']=np.array(morph_flag)
ceers_cat['delta_f444']=np.array(delta_value)


morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_f150w_mean,ceers_cat.disk_f150w_mean,ceers_cat.irr_f150w_mean,ceers_cat.bd_f150w_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f150w']=np.array(morph_flag)
ceers_cat['delta_f150']=np.array(delta_value)


logf200 = np.log10(ceers_cat['FLUX_200'].values)
logA = np.log10(ceers_cat['A_IMAGE'].values)
class_star = ceers_cat.CLASS_STAR_200

star_flag = logf200*0.0
delta_loc = logA-0.2*logf200+0.01
star_flag[delta_loc<0]=1
ceers_cat['star_flag']=star_flag

#ceers_cat.to_csv(data_path+"CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv")




def build_features(ceers_cat,zbin,node_features):
    x = node_features['x']   # halo mass and concentration
    t = node_features['t']  

    x_updated=[]
    t_updated=[]

    for x_data, t_data in zip(x,t):
        #print(x_data)

        sel2 = ceers_cat.query("F356W_RE > 0  and logM_50>7 and zfit_50>"+str(zbin[0])+" and zfit_50<"+str(zbin[1]))
        #print(len(sel2))
        for i in range(len(sel2)):
            #print(sel2.logM_50.values[i]-np.log10(2)-x_data[-1,0])
            # Append 'x_data' and 't_data' for the current galaxy in the second bin
            if (sel2.logM_50.values[i]-np.log10(2)-x_data[-1,0])>0.5 or (sel2.logM_50.values[i]-np.log10(2)-x_data[-1,0])<-3:
                continue

               
            new_entry  = [sel2.logM_50.values[i]-np.log10(2),np.log10(0.8*Planck15.angular_diameter_distance(sel2.zfit_50.values[i]).value * np.deg2rad(sel2.F356W_RE.values[i] / 3600) * 1e3)]
            new_x = np.vstack([x_data, new_entry])

            new_entry = [1/(1+sel2.zfit_50.values[i]) ] 
            new_t = np.vstack([t_data,new_entry])
            #pdb.set_trace()

            # Convert the 'x_data' to a list of floats while ignoring non-numeric and 'inf' values and skipping the first row
            cleaned_x_mass = [float(value) for value,size in zip(new_x[0:,0],new_x[0:,1]) if value != b'-' and value != b'-inf']
            cleaned_x_size = [float(value) for value in (new_x[0:,1]) if value != b'-' and value != b'-inf']
            x_copy = np.column_stack([cleaned_x_mass, cleaned_x_size])

            cleaned_t = [float(value) for value,size in zip(new_t[0:,0],new_x[0:,1]) if value != b'-' and value != b'-inf']
            cleaned_t = np.expand_dims(cleaned_t,1)

            x_updated.append(x_copy)
            t_updated.append(cleaned_t)

            #pdb.set_trace()

      # Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
    node_features = {'x': [np.array(arr, dtype=np.float32) for arr in x_updated], 't': [np.array(arr, dtype=np.float32) for arr in t_updated]}
    return node_features, len(sel2)
           






def build_roots(ceers_cat,zbins=[0,.75,1.5]):
    


    sel = ceers_cat.query("F356W_RE > 0 and logM_50>9.5 and zfit_50>"+str(zbins[0])+" and zfit_50<"+str(zbins[1]))

    x = []
    t = []
    node_features = {'x': None, 't': None}
    for mass,z,re in zip(sel.logM_50,sel.zfit_50,sel.F356W_RE):

        x_data=np.zeros((2,2))
        t_data=np.zeros(2)
        x_data[0,0]=mass-np.log10(2)

        x_data[0,1]=0.8*Planck15.angular_diameter_distance(z).value * np.deg2rad(re / 3600) * 1e3
        t_data[0] = 1/(1+z)   

        sel2 =  ceers_cat.query("F356W_RE > 0  and logM_50>7 and zfit_50>"+str(zbins[1])+" and zfit_50<"+str(zbins[2]))
        
        for i in range(len(sel2)):
            #print(x_data[0,0])
            #print((sel2.logM_50.values[i]-np.log10(2)-x_data[0,0]))
            # Append 'x_data' and 't_data' for the current galaxy in the second bin
            if (sel2.logM_50.values[i]-np.log10(2)-x_data[0,0])>0.5 or (sel2.logM_50.values[i]-np.log10(2)-x_data[0,0])<-3:
                continue

            #print('here')     
            x_data[1,0]=sel2.logM_50.values[i]-np.log10(2)
            x_data[1,1]=0.8*Planck15.angular_diameter_distance(sel2.zfit_50.values[i]).value * np.deg2rad(sel2.F356W_RE.values[i] / 3600) * 1e3
            t_data[1]=1/(1+sel2.zfit_50.values[i])  

            # Convert the 'x_data' to a list of floats while ignoring non-numeric and 'inf' values and skipping the first row
            cleaned_x_mass = [float(value) for value,size in zip(x_data[0:,0],x_data[0:,1]) if value != b'-' and value != b'-inf']
            cleaned_x_size = [float(value) for value in (x_data[0:,1]) if value != b'-' and value != b'-inf']

            #print(np.array(cleaned_x_mass).shape)
            x_copy = np.column_stack([cleaned_x_mass, np.log10(cleaned_x_size)])
            #print(np.array(x_copy).shape)
            #cleaned_x = [float(value) for value in x_data[1:] if value != b'-' and value != b'-inf']
            # Convert the 't_data' to a list of floats while ignoring non-numeric and 'inf' values and skipping the first row
            cleaned_t = [float(value) for value,size in zip(t_data[0:],x_data[0:,1]) if value != b'-' and value != b'-inf']
            cleaned_t = np.expand_dims(cleaned_t,1)
            
            # Append the cleaned 'x' and 't' data to their respective lists
            x.append(x_copy)
            t.append(cleaned_t)





    # Store 'x_copy' and 't' data as lists of NumPy arrays in the 'node_features' dictionary
    node_features = {'x': [np.array(arr, dtype=np.float32) for arr in x], 't': [np.array(arr, dtype=np.float32) for arr in t]}
    return node_features,len(sel2)

    



def get_maxlike_descendant(l_numpy,node_features, chunk_size,step=1):

    x=node_features['x']
    t=node_features['t']
    # Calculate the number of chunks
    
    
    x_sel=[]
    t_sel=[]
    if step>1:
        l_numpy = l_numpy[1::step]
    num_chunks = len(l_numpy) // chunk_size    
    print(num_chunks)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = l_numpy[start_idx:end_idx]

        # Find the maximum value and its position in the chunk
        max_value = np.max(chunk)
        max_position = np.argmax(chunk)

        x_sel.append(x[start_idx + max_position])
        t_sel.append(t[start_idx + max_position])
        
    node_features_updated = {'x': [np.array(arr, dtype=np.float32) for arr in x_sel], 't': [np.array(arr, dtype=np.float32) for arr in t_sel]}
    return node_features_updated


import torch
from typing import Optional, Tuple
from torch import Tensor
def log_likelihood_obs(
        model: torch.nn.Module, batch: Tuple[Tensor],
        to_numpy: bool = True,  batch_size: int = 4096
    ):
    """ Sample trees using Recurrent-MAF model
    Parameters
    ----------
    model: torch.nn.Module
        Recurrent model
    roots: np.ndarray
        Root features
    times: np.ndarray
        Time features
    to_numpy: bool
        Whether to convert to numpy
    device: Optional
        Device to use
    batch_size: int

    Returns
    -------
    x: Union[Tensor, np.ndarray]
        Sampled trees
    """

    device ='cpu'
    model = model.to(device)

    #seq_len = len(roots)+1
    #mask = np.expand_dims(np.zeros(seq_len),axis=0)
    #mask = np.zeros((1, seq_len), dtype=np.bool)
    #mask[:, :seq_len] = True
    #t = np.concatenate((times, t_obs),axis=1)

    #x_tensor = torch.from_numpy(roots.astype('float32'))
    #y_tensor = torch.from_numpy(obs.astype('float32'))
    #t_tensor = torch.from_numpy(t.astype('float32'))
    #seq_len_tensor = torch.tensor(seq_len, dtype=torch.int32)
    #mask_tensor = torch.from_numpy(mask)

    #tensor_tuple = (x_tensor, y_tensor, t_tensor, seq_len_tensor, mask_tensor)

    lp=model.log_prob(batch,return_context=False)

   
    return lp    

# Load the trained model from a checkpoint file

checkpoint_path = "/scratch/mhuertas/CEERS/proj/TNGEagle_mass_size_gt9/last.ckpt"  # Specify the path to your checkpoint file
loaded_model = DataModule.load_from_checkpoint(checkpoint_path)

# Set the model to evaluation mode (important if you have dropout or batch normalization layers)
loaded_model.eval()



#redshifts = np.array([1.5,2,2.5,3,4,6])
redshifts = np.array([1.5,2,2.5,3.5])

node_features,chunk_size = build_roots(ceers_cat)
preprocessed_node_features = loaded_model.transform(node_features, fit=False)
l  = log_likelihood_obs(loaded_model,preprocessed_node_features)
l_numpy = l.detach().numpy()
node_features = get_maxlike_descendant(l.detach().numpy(),node_features,chunk_size)

step=2
for zmin,zmax in zip(redshifts[:-1],redshifts[1:]):
    print(zmin,zmax)
    node_features,chunk_size = build_features(ceers_cat,[zmin,zmax],node_features)
    preprocessed_node_features = loaded_model.transform(node_features, fit=False)
    l  = log_likelihood_obs(loaded_model,preprocessed_node_features)
    node_features = get_maxlike_descendant(l.detach().numpy(),node_features, chunk_size,step=step)
    step+=1
    
