import pandas as pd
import numpy as np
from astropy.io import fits

data_path="/Users/marchuertascompany/Documents/data/FORECAST/"
cat_filename ="final_cat.jwst.cb16.csv"
part_filename="planes_list_z.dat.csv"

forecast = pd.read_csv(data_path+cat_filename)
part = pd.read_csv(data_path+part_filename)


z_i = part.z_i
z_iplus1=part["z_i+1"]
snap = part.TNG_sn

# Initialize empty lists to store snap_vec and shid_vec
snap_vec_list = []
shid_vec_list = []

for zmin,zmax,sn in zip(z_i,z_iplus1,snap):
    sel = forecast.query('z>='+str(zmin)+' and z<='+str(zmax))
    snap_vec = (np.zeros(len(sel))+sn).astype('int32')
    shid_vec = sel.shID

    # Append snap_vec and shid_vec to the respective lists
    snap_vec_list.extend(snap_vec)
    shid_vec_list.extend(shid_vec)

# Create a DataFrame with snap_vec and shid_vec columns
result_df = pd.DataFrame({'snap_vec': snap_vec_list, 'shid_vec': shid_vec_list})

# Save the DataFrame to a CSV file
result_df.to_csv(data_path+'snap_shid_vectors.csv', index=False)








