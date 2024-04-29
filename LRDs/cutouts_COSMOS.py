import pandas as pd
import numpy as np

data_path = "/n03data/huertas/COSMOS-Web/cats/"
# Load your DataFrame
ceers_cat = pd.read_csv(data_path+'COSMOSWeb_master_v1.6.0-sersic+BD-em_cgs_LePhare_nodupl_nomulti_morph_F150W_F277W_F444W.csv')  # Replace this with your actual file path


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


#ceers_cat.to_csv(data_path+"CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv")


# Define your queries as a list of tuples (query, number_of_ids)
# Each tuple consists of a query string and the number of ids you want to select for that query



queries = [
    ("FLAG_STAR_JWST==0 and morph_flag_f150w==0 and MAG_MODEL_F150W<24.5", 20,'sph','f150w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f150w==1 and MAG_MODEL_F150W<24.5", 20,'disk','f150w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f150w==2 and MAG_MODEL_F150W<24.5", 20,'irr','f150w'),
    ("FLAG_STAR_JWST==0 and morph_flag_f150w==3 and MAG_MODEL_F150W<24.5", 20,'bd','f150w'),
    ("FLAG_STAR_JWST==0 and morph_flag_f277w==0 and MAG_MODEL_F277W<24.5", 20,'sph','f277w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f277w==1 and MAG_MODEL_F277W<24.5", 20,'disk','f277w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f277w==2 and MAG_MODEL_F277W<24.5", 20,'irr','f277w'),
    ("FLAG_STAR_JWST==0 and morph_flag_f277w==3 and MAG_MODEL_F277W<24.5", 20,'bd','f277w'),
    ("FLAG_STAR_JWST==0 and morph_flag_f444w==0 and MAG_MODEL_F444W<24.5", 20,'sph','f444w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f444w==1 and MAG_MODEL_F444W<24.5", 20,'disk','f444w'),  # Replace with your actual query and number of ids
    ("FLAG_STAR_JWST==0 and morph_flag_f444w==2 and MAG_MODEL_F444W<24.5", 20,'irr','f444w'),
    ("FLAG_STAR_JWST==0 and morph_flag_f444w==3 and MAG_MODEL_F444W<24.5", 20,'bd','f444w')
    # Add more queries as needed
]


#142965 150.350119   2.002229
#255808 150.033133   2.016092
#224295 149.938429   2.060233
#189506 149.807416   2.086350
# 25183 149.907771   2.249254
#179548 149.811418   2.022600
# 73839 150.127293   2.071421
# 35215 149.890717   2.125640
#217297 149.933428   2.016641
#182535 149.711431   2.078097
# 67493 150.137782   2.023508
# 59170 150.015253   2.233939

# Base command parts
base_command = "python /home/huertas/python/makeCOSMOScutouts/makeCutouts-Full-CW.py"
output_path = "/n03data/huertas/COSMOS-Web/cutouts/"
cutout_size = 2.0

# Open the shell script file
with open("/n03data/huertas/COSMOS-Web/cutouts/generate_cutouts.sh", "w") as file:
    file.write("#!/bin/bash\n")

    
    # Filter the DataFrame according to the query, select random ids, then sort them in descending order
    #filtered_ids = [142965,255808,224295,189506,25183,179548,73839,35215,217297,182535,67493,59170] # Change random_state for different subsets
    filtered_ids = [458207,571082,539569,504780,340060,494822,388983,350232,532571,497809,382637,374187]

    # Create the command with selected ids
    source_ids_str = ' '.join(map(str, filtered_ids))
    command = f"{base_command} --source_ids {source_ids_str} --output_path {output_path + 'LRDs_bulges_z2-3'} --cutout_size {cutout_size}\n"

        
    # Write the command to the file
    file.write(command)

print("Shell script 'generate_cutouts.sh' has been created with multiple commands.")
