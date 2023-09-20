'''
Y. Dong, Sept 6
test opening the CEERS catalog
'''

import pandas as pd
import os

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"


cat = pd.read_csv(os.path.join(cat_dir,cat_name),nrows=10100)
col_names = ['RA_1','DEC_1']
cols = cat.columns

for col in cols:
    print(col)

for i in range(100):
    print(cat[['ID','RA_1','DEC_1','RA_2','DEC_2']].values[i])
