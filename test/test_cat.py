'''
Y. Dong, Sept 6
test opening the CEERS catalog
'''

import pandas as pd
import os

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"


cat = pd.read_csv(os.path.join(cat_dir,cat_name),nrows=10100)

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cla = pd.read_csv(os.path.join(class_dir,class_name))

col_names = ['RA_1','DEC_1']
cols = cla.columns

for col in cols:
    print(col)

print(cla['mag_select'])

for i in range(100):
    print(cat[['ID','RA_1','DEC_1','RA_2','DEC_2']].values[i])
