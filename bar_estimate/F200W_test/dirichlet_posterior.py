import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import betabinom
from scipy.special import gamma, beta

def str2array(string):

    numbers_str = string.strip('[]').split(',')
    array = np.array([float(num) for num in numbers_str])
    return array


def generalized_beta_binomial_pmf(x, n:int, a, b):

    return gamma(n+1)/gamma(x+1)/gamma(n-x+1)*beta(x+a,n-x+b)/beta(a,b)


# load catalog
cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name))
z = cat['zfit_50'].values


alpha_feature_pred = []
alpha_smooth_pred = []
alpha_artifact_pred = []

alpha_strong_pred = []
alpha_weak_pred = []
alpha_none_pred = []

for i in range(3):
    # load the finetuned Zoobot predictions 
    pred_path = f"bar_estimate/F200W_pred/full_cat_predictions_F200W_{i}.csv"
    pred = pd.read_csv(pred_path)

    # pred = pred[pred['id_str']<20000]
    id = pred['id_str'].values
    print(len(id))
    # np.sort(id)
    print(id[:300])

    alpha_feature_pred.append(pred['t0_smooth_or_featured__features_or_disk_pred'].values)
    alpha_smooth_pred.append(pred['t0_smooth_or_featured__smooth_pred'].values)
    alpha_artifact_pred.append(pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values)
    # p_feature_pred = alpha_feature_pred/(alpha_feature_pred+alpha_smooth_pred+alpha_artifact_pred)

    alpha_strong_pred.append(pred['t4_is_there_a_bar__strong_bar_pred'].values)
    alpha_weak_pred.append(pred['t4_is_there_a_bar__weak_bar_pred'].values)
    alpha_none_pred.append(pred['t4_is_there_a_bar__no_bar_pred'].values)
    # p_strong_pred = alpha_strong_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)
    # p_weak_pred = alpha_weak_pred/(alpha_strong_pred+alpha_weak_pred+alpha_none_pred)

    # p_bar_pred = p_strong_pred+p_weak_pred


match_path = "bot/match_catalog_F200W.csv"
match_cat = pd.read_csv(match_path)
match_id = match_cat['id_str'].values
image_loc = match_cat['file_loc'].values

count_feature_vol = match_cat['t0_smooth_or_featured__features_or_disk'].values
count_smooth_vol = match_cat['t0_smooth_or_featured__smooth'].values
count_artifact_vol = match_cat['t0_smooth_or_featured__star_artifact_or_bad_zoom'].values
total_feature_question_vol = count_feature_vol+count_smooth_vol+count_artifact_vol
p_feature_vol = count_feature_vol/(count_feature_vol+count_smooth_vol+count_artifact_vol)

count_strong_vol = match_cat['t4_is_there_a_bar__strong_bar'].values
count_weak_vol = match_cat['t4_is_there_a_bar__weak_bar'].values
count_none_vol = match_cat['t4_is_there_a_bar__no_bar'].values
total_bar_question_vol = count_strong_vol+count_weak_vol+count_none_vol
p_strong_vol = count_strong_vol/(count_strong_vol+count_weak_vol+count_none_vol)
p_weak_vol = count_weak_vol/(count_strong_vol+count_weak_vol+count_none_vol)

p_bar_vol = p_strong_vol+p_weak_vol


common_id, index1, index2 = np.intersect1d(id, match_id, return_indices=True)


for i in np.random.randint(len(id), size=20):
# for i in np.where((alpha_strong_pred + alpha_weak_pred) > 2*alpha_none_pred)[0]:
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.imshow(mpimg.imread(image_loc[index2[i]]), cmap='gray')
    plt.title('F200W image')

    n1 = int(total_feature_question_vol[index2[i]])
    x1 = np.linspace(0, n1, 100*n1)

    a_feature = str2array(alpha_feature_pred[i])
    b_feature = str2array(alpha_smooth_pred[i]) + str2array(alpha_artifact_pred[i])
    pdf_feature = np.zeros((5, 100*n1))

    plt.subplot(1, 3, 2)
    plt.xlim((0,n1))
    plt.ylim(bottom=0)
    for j in range(5):
        pdf_feature[j,:] = generalized_beta_binomial_pmf(x1, n1, a_feature[j], b_feature[j])
        plt.plot(x1, pdf_feature[j,:], color='orange', alpha=0.3)
    plt.plot(x1, np.mean(pdf_feature, axis=0), color='blue', alpha=1)
    plt.axvline(p_feature_vol[index2[i]]*n1, color='black', linestyle='dashed')
    plt.title(r'Feature votes')

    n2 = int(total_bar_question_vol[index2[i]])
    x2 = np.linspace(0, n2, 100*n2)

    a_bar = str2array(alpha_strong_pred[i]) + str2array(alpha_weak_pred[i])
    b_bar = str2array(alpha_none_pred[i])
    pdf_bar = np.zeros((5, 100*n2))

    plt.subplot(1, 3, 3)
    plt.xlim((0,n2))
    plt.ylim(bottom=0)
    for j in range(5):
        pdf_bar[j,:] = generalized_beta_binomial_pmf(x2, n2, a_bar[j], b_bar[j])
        plt.plot(x2, pdf_bar[j,:], color='orange', alpha=0.3)
    plt.plot(x2, np.mean(pdf_bar, axis=0), color='blue', alpha=1)
    plt.axvline(p_bar_vol[index2[i]]*n2, color='black', linestyle='dashed')
    plt.title(r'Bar votes')

    plt.tight_layout()
    plt.savefig(f'bar_estimate/F200W_test/F200W_dirichlet_id{id[i]}.png')

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