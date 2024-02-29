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
    ("morph_flag_f150w==0 and MAG_MODEL_F150W<24.5", 20,'sph','f150w'),  # Replace with your actual query and number of ids
    ("morph_flag_f150w==1 and MAG_MODEL_F150W<24.5", 20,'disk','f150w'),  # Replace with your actual query and number of ids
    ("morph_flag_f150w==2 and MAG_MODEL_F150W<24.5", 20,'irr','f150w'),
    ("morph_flag_f150w==3 and MAG_MODEL_F150W<24.5", 20,'bd','f150w'),
    ("morph_flag_f277w==0 and MAG_MODEL_F277W<24.5", 20,'sph','f277w'),  # Replace with your actual query and number of ids
    ("morph_flag_f277w==1 and MAG_MODEL_F277W<24.5", 20,'disk','f277w'),  # Replace with your actual query and number of ids
    ("morph_flag_f277w==2 and MAG_MODEL_F277W<24.5", 20,'irr','f277w'),
    ("morph_flag_f277w==3 and MAG_MODEL_F277W<24.5", 20,'bd'),'f277w',
    ("morph_flag_f444w==0 and MAG_MODEL_F444W<24.5", 20,'sph','f444w'),  # Replace with your actual query and number of ids
    ("morph_flag_f444w==1 and MAG_MODEL_F444W<24.5", 20,'disk','f444w'),  # Replace with your actual query and number of ids
    ("morph_flag_f444w==2 and MAG_MODEL_F444W<24.5", 20,'irr','f444w'),
    ("morph_flag_f444w==3 and MAG_MODEL_F444W<24.5", 20,'bd','f444w')
    # Add more queries as needed
]

# Base command parts
base_command = "python /home/shuntov/COSMOS-Web/TheSurvey/makeCutouts.py"
output_path = "/n03data/huertas/COSMOS-Web/cutouts/"
cutout_size = 2.0

# Open the shell script file
with open("/n03data/huertas/COSMOS-Web/cutouts/generate_cutouts.sh", "w") as file:
    file.write("#!/bin/bash\n")

    # Process each query
    for query, n,m,f in queries:
        # Filter the DataFrame according to the query, select random ids, then sort them in descending order
        filtered_ids = ceers_cat.query(query)['Id_1'].sample(n=n, random_state=1).sort_values(ascending=False)  # Change random_state for different subsets
        
        # Create the command with selected ids
        source_ids_str = ' '.join(map(str, filtered_ids))
        command = f"{base_command} --source_ids {source_ids_str} --output_path {output_path+f+m} --cutout_size {cutout_size}\n"
        
        # Write the command to the file
        file.write(command)

print("Shell script 'generate_cutouts.sh' has been created with multiple commands.")
