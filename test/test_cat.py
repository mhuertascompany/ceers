'''
Y. Dong, Sept 6
test opening the CEERS catalog
'''

import pandas as pd
import os
import numpy as np

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"


cat = pd.read_csv(os.path.join(cat_dir,cat_name),nrows=10100)

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cla = pd.read_csv(os.path.join(class_dir,class_name))

col_names = ['RA_1','DEC_1']
cols = cat.columns

# for col in cols:
#     print(col)

# print(cla[['pixrad','radius_select','flux_rad_0p50','which_nircam','nircam_id']].values)
print(np.unique(cla['which_nircam'].values))

for i in range(100,200):
    print(cat[['ID','zfit_50','zfit_16','zfit_84','logM_50','logM_16','logM_84','logMt_50','logMt_16','logMt_84']].values[i])
