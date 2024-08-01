import os
import pandas as pd
from astropy.table import Table
import numpy as np

cat_dir = "/n03data/huertas/COSMOS-Web/cats"
filename = 'merged_catalog_samples.csv'
ceers_cat=pd.read_csv(os.path.join(cat_dir,filename))




filters = ['F150W','F277W','F444W']
morph=['sph','disk','irr','bd']

for f in filters:
    for m in morph:
        c = ceers_cat.filter(regex='^'+m+'_')
        c = c.filter(regex=f+'$')
        ceers_cat[m+'_'+f+'_mean']=c.mean(axis=1).values
        ceers_cat[m+'_'+f+'_std']=c.std(axis=1).values







morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_F150W_mean,ceers_cat.disk_F150W_mean,ceers_cat.irr_F150W_mean,ceers_cat.bd_F150W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f150w']=np.array(morph_flag)
ceers_cat['delta_f150']=np.array(delta_value)

morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_F277W_mean,ceers_cat.disk_F277W_mean,ceers_cat.irr_F277W_mean,ceers_cat.bd_F277W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f277w']=np.array(morph_flag)
ceers_cat['delta_f277']=np.array(delta_value)

morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(ceers_cat.sph_F444W_mean,ceers_cat.disk_F444W_mean,ceers_cat.irr_F444W_mean,ceers_cat.bd_F444W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f444w']=np.array(morph_flag)
ceers_cat['delta_f444']=np.array(delta_value)



queries = [
    ("LP_mass_med_PDF>11 and LP_zfinal>2 and r_kpc>9", 10,'bw','f277w'),  # Replace with your actual query and number of ids

    # Add more queries as needed
]



# Base command parts
base_command = "python /n03data/huertas/python/makeCOSMOScutouts/makeCutouts.py"
output_path = "/n03data/huertas/COSMOS-Web/cutouts/"
cutout_size = 2.0

# Open the shell script file
with open("/n03data/huertas/COSMOS-Web/cutouts/generate_cutouts_BW.sh", "w") as file:
    file.write("#!/bin/bash\n")

    # Process each query
    for query, n,m,f in queries:
        # Filter the DataFrame according to the query, select random ids, then sort them in descending order
        filtered_ids = ceers_cat.query(query)['Id'].sample(n=n, random_state=1).sort_values(ascending=False)  # Change random_state for different subsets
        
        # Create the command with selected ids
        source_ids_str = ' '.join(map(str, filtered_ids))
        command = f"{base_command} --source_ids {source_ids_str} --output_path {output_path+f+m} --cutout_size {cutout_size}\n"
        
        # Write the command to the file
        file.write(command)

print("Shell script 'generate_cutouts.sh' has been created with multiple commands.")