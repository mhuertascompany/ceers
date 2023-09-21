'''
Y. Dong, Sept 20
test opening the GZ CEERS classifications
'''

import pandas as pd
import os
import numpy as np
import re
# import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord

class_dir = "/scratch/ydong/classifications"
class_name = "jwst-ceers-v0-5-aggregated-class-singlechoicequestionsonly.csv"

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

image_dir = '/scratch/ydong/stamps/demo_F200W'
file_loc = [os.path.join(image_dir,path) for path in os.listdir(image_dir)]
ids = np.array([int(re.findall(r'\d+',path)[1]) for path in os.listdir(image_dir)])

cla = pd.read_csv(os.path.join(class_dir,class_name))
cat = pd.read_csv(os.path.join(cat_dir,cat_name)).iloc[ids]


cols = cla.columns

for col in cols:
    print(repr(col))
# col_names = ['RA_1','DEC_1']

N = len(cla)
match_num = np.zeros(N).astype(int)

cat_ra = np.round(cat['RA_1'].values,6)
cat_dec = np.round(cat['DEC_1'].values,6)
cla_ra = np.round(cla['RA'].values,6)
cla_dec = np.round(cla['Dec'].values,6)

c = SkyCoord(ra=cla_ra*u.degree, dec=cla_dec*u.degree)

catalog = SkyCoord(ra=cat_ra*u.degree, dec=cat_dec*u.degree)

idx, d2d, d3d = c.match_to_catalog_sky(catalog)

cat2class = -1*np.ones(len(ids)).astype(int)

mask = d2d<0.5*u.arcsec

for i in range(N):
    if mask[i]:
        if cat2class[idx[i]] >= 0:
            prev_match = cat2class[idx[i]]
            if d2d[i] < d2d[prev_match]:
                cat2class[idx[i]] = i
                mask[prev_match] = False
            else: 
                mask[i] = False
        else:
            cat2class[idx[i]] = i
        
print("Total matches: %i"%np.sum(mask))


gz_answers = [
    't0_smooth_or_featured__features_or_disk',
    't0_smooth_or_featured__smooth',
    't0_smooth_or_featured__star_artifact_or_bad_zoom',
    't1_how_rounded_is_it__cigarshaped',
    't1_how_rounded_is_it__in_between',
    't1_how_rounded_is_it__completely_round',
    't2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk',
    't2_could_this_be_a_disk_viewed_edgeon__no_something_else',
    't3_edge_on_bulge_what_shape__boxy',
    't3_edge_on_bulge_what_shape__no_bulge',
    't3_edge_on_bulge_what_shape__rounded',
    't4_is_there_a_bar__no_bar',
    't4_is_there_a_bar__strong_bar',
    't4_is_there_a_bar__weak_bar',
    't5_is_there_any_spiral_arm_pattern__yes',
    't5_is_there_any_spiral_arm_pattern__no',
    't6_spiral_how_tightly_wound__loose',
    't6_spiral_how_tightly_wound__medium',
    't6_spiral_how_tightly_wound__tight',
    't7_how_many_spiral_arms_are_there__1',
    't7_how_many_spiral_arms_are_there__2',
    't7_how_many_spiral_arms_are_there__3',
    't7_how_many_spiral_arms_are_there__4',
    't7_how_many_spiral_arms_are_there__more_than_4',
    't7_how_many_spiral_arms_are_there__cant_tell',
    't8_not_edge_on_bulge__dominant',
    't8_not_edge_on_bulge__moderate',
    't8_not_edge_on_bulge__no_bulge',
    't8_not_edge_on_bulge__large',
    't8_not_edge_on_bulge__small',
    't11_is_the_galaxy_merging_or_disturbed__major_disturbance',
    't11_is_the_galaxy_merging_or_disturbed__merging',
    't11_is_the_galaxy_merging_or_disturbed__minor_disturbance',
    't11_is_the_galaxy_merging_or_disturbed__none',
    't12_are_there_any_obvious_bright_clumps__yes',
    't12_are_there_any_obvious_bright_clumps__no',
    't19_what_problem_do___e_with_the_image__nonstar_artifact',
    't19_what_problem_do___e_with_the_image__bad_image_zoom',
    't19_what_problem_do___e_with_the_image__star'
]

gz_counts = [a+'__count' for a in gz_answers]
match_catalog = cla.loc[mask, gz_counts]
match_catalog.columns = [col[:-7] for col in match_catalog.columns]
match_catalog['id_str'] = ids[idx[mask]]
match_catalog['file_loc'] = [file_loc[k] for k in idx[mask]]

match_catalog.to_csv("bot/match_catalog_F200W.csv")

# for col in cols:
#     print(col)

# print("0 match: %i"%np.sum(match_num==0))
# print("1 match: %i"%np.sum(match_num==1))
# print("more matches: %i"%np.sum(match_num>=2))

# plt.hist(match_num)
# plt.savefig('test/hist.png')
