'''
Y. Dong, Sept 6
test opening the CEERS catalog
'''

import pandas as pd
import os

cat_dir = "/scratch/mhuertas/CEERS/data_release/cats"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v051_bug.csv"


cat = pd.read_csv(os.path.join(cat_dir,cat_name),nrows=10)
col_names = ['RA_1','DEC_1']

print(type(cat[col_names]))
