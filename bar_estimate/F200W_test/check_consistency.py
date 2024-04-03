import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# load catalog
cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))
z = cat['zfit_50'].values

# directory for images
image_dir1 = "/scratch/ydong/stamps/demo_F200W"
image_dir2 = "/scratch/ydong/stamps/demo_F200W_added"

# load the finetuned Zoobot predictions 
pred_path = "results/finetune_tree_result/F200W/demo_tree_predictions_F200W_1.csv"
pred = pd.read_csv(pred_path)

id = pred['id_str'].values

alpha_feature_pred = pred['t0_smooth_or_featured__features_or_disk_pred'].values
alpha_smooth_pred = pred['t0_smooth_or_featured__smooth_pred'].values
alpha_artifact_pred = pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values
p_feature_pred = alpha_feature_pred/(alpha_feature_pred+alpha_smooth_pred+alpha_artifact_pred)

alpha_strong_pred = pred['t4_is_there_a_bar__strong_bar_pred'].values
alpha_weak_pred = pred['t4_is_there_a_bar__weak_bar_pred'].values
alpha_none_pred = pred['t4_is_there_a_bar__no_bar_pred'].values
p_strong_pred = alpha_strong_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)
p_weak_pred = alpha_weak_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)

p_bar_pred = p_strong_pred+p_weak_pred


match_path = "bot/match_catalog_F200W.csv"
match_cat = pd.read_csv(match_path)
match_id = match_cat['id_str'].values

count_feature_vol = match_cat['t0_smooth_or_featured__features_or_disk'].values
count_smooth_vol = match_cat['t0_smooth_or_featured__smooth'].values
count_artifact_vol = match_cat['t0_smooth_or_featured__star_artifact_or_bad_zoom'].values
p_feature_vol = count_feature_vol/(count_feature_vol+count_smooth_vol+count_artifact_vol)

count_strong_vol = match_cat['t4_is_there_a_bar__strong_bar'].values
count_weak_vol = match_cat['t4_is_there_a_bar__weak_bar'].values
count_none_vol = match_cat['t4_is_there_a_bar__no_bar'].values
p_strong_vol = count_strong_vol/(count_strong_vol+count_weak_vol+count_none_vol)
p_weak_vol = count_weak_vol/(count_strong_vol+count_weak_vol+count_none_vol)

p_bar_vol = p_strong_vol+p_weak_vol


common_id, index1, index2 = np.intersect1d(id, match_id, return_indices=True)

plt.figure(figsize=(5,5))
plt.xlim((0,1))
plt.ylim((0,1))
plt.scatter(p_feature_pred,p_feature_vol[index2],s=2)
plt.savefig("bar_estimate/F200W_test/p_feature_consistency.jpg")

plt.figure(figsize=(5,5))
plt.xlim((0,1))
plt.ylim((0,1))
plt.scatter(p_bar_pred,p_bar_vol[index2],s=2)
plt.savefig("bar_estimate/F200W_test/p_bar_consistency.jpg")