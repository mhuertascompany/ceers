'''
Y. Dong, Sept 20
test opening the GZ CEERS classifications
'''

import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

image_dir = '/scratch/ydong/stamps/demo_F200W'
ids = [int(re.findall(r'\d+',path)[1]) for path in os.listdir(image_dir)]

cla = pd.read_csv(os.path.join(class_dir,class_name))
cat = pd.read_csv(os.path.join(cat_dir,cat_name))


cols = cla.columns

for col in cols:
    print(col)
# col_names = ['RA_1','DEC_1']
N = len(cla)
match_num = np.zeros(N).astype(int)

cat_ra = np.round(cat['RA_1'].values[ids],6)
cat_dec = np.round(cat['DEC_1'].values[ids],6)
cla_ra = np.round(cla['RA'].values,6)
cla_dec = np.round(cla['Dec'].values,6)

c = SkyCoord(ra=cla_ra*u.degree, dec=cla_dec*u.degree)

catalog = SkyCoord(ra=cat_ra*u.degree, dec=cat_dec*u.degree)

idx, d2d, d3d = c.match_to_catalog_sky(catalog)

mask = d2d<0.2*u.arcsec
print(np.sum(mask))
for i in range(1000):
    if mask[i]:
        print(i,ids[idx[i]],d2d[i])
        print(i,cla[['t0_smooth_or_featured__features_or_disk__frac','t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk__frac','t4_is_there_a_bar__no_bar__frac']].values[i])

# for col in cols:
#     print(col)
for i in range(300,500):
    for j in ids:
        if (cat_ra[j]==cla_ra[i]) & (cat_dec[j]==cla_dec[i]):
            match_num[i] += 1
            print(i,cla[['t0_smooth_or_featured__features_or_disk__frac','t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk__frac','t4_is_there_a_bar__no_bar__frac']].values[i])
            print(j)

print("0 match: %i"%np.sum(match_num==0))
print("1 match: %i"%np.sum(match_num==1))
print("more matches: %i"%np.sum(match_num>=2))

plt.hist(match_num)
plt.savefig('test/hist.png')
