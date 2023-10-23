'''
Using incremental PCA and umap to visualize the representations of CEERS F200W dataset, 
as in M. Walmsley's 2022 paper
'''

import numpy as np
import pandas as pd
import os
import umap
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import IncrementalPCA

from zoobot.shared import load_predictions

time_start = time.time()

repr_loc = 'finetune_repr/repr_results/F200W_representations.hdf5'
repr_df = load_predictions.single_forward_pass_hdf5s_to_df(repr_loc)
feat_cols = [f'feat_{n}' for n in range(1280)]
repr = repr_df[feat_cols].values
id_str = repr_df['id_str'].values.astype(int)

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))
mag = cat['F200W_MAG'].values
sersic = cat['F200W_N'].values

col_list = ['RA_1','logSFRinst_50','F200W_Q','F200W_PA','F200W_RE','F200W_N']
morph_cols = ['sph_f150w_mean','disk_f150w_mean','irr_f150w_mean','bd_f150w_mean']
# morph = np.array([['sph','disk','irr','bd'][morphclass] for morphclass in np.argmax(cat[morph_cols].values,axis=1)])

N_COMP = 15 # number of components remaining after IPCA
ipca = IncrementalPCA(n_components=N_COMP)
repr_pca = ipca.fit_transform(repr)

reducer = umap.UMAP()
embedding = reducer.fit_transform(repr_pca)
dim1 = embedding[:,0]
dim2 = embedding[:,1]
N_SAM = embedding.shape[0]

plt.figure(figsize=(15,10))
# colors = sns.color_palette('viridis', n_colors=4)
# scatter = sns.scatterplot(
#     x=dim1,y=dim2,
#     hue=morph[id_str],
#     palette=colors)
# scatter.set_title('UMAP F200W morphology classes')
# scatter.set_xlabel('UMAP 1')
# scatter.set_ylabel('UMAP 2')
# scatter.legend(scatterpoints=1, markerscale=1)
plt.scatter(dim1,dim2,c=np.log(sersic[id_str]),cmap='RdBu')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.title(f"UMAP F200W color coded by log F200W_N")
plt.savefig(f'finetune_repr/repr_results/F200W_logF200W_N_coded_new.png')

size = 1.   # size of one single thumbnail
min1 = dim1.min()
max1 = dim1.max()
mid1 = (min1+max1)/2
min2 = dim2.min()
max2 = dim2.max()
mid2 = (min2+max2)/2
n1 = np.ceil((max1-min1)/size).astype(int)
n2 = np.ceil((max2-min2)/size).astype(int)
grid1 = np.linspace(mid1-n1*size/2,mid1+n1*size/2,n1+1)
grid2 = np.linspace(mid2-n2*size/2,mid2+n2*size/2,n2+1)


plt.figure(figsize=(20,20))
plt.xlim(mid1-n1*size/2,mid1+n1*size/2)
plt.ylim(mid2-n2*size/2,mid2+n2*size/2)

for i in range(n1):
    for j in range(n2):
        in_region = (dim1>grid1[i]) & (dim1<grid1[i+1]) & (dim2>grid2[j]) & (dim2<grid2[j+1])
        if np.any(in_region):
            all_local_ids = id_str[in_region]
            tn_id = all_local_ids[np.argmin(mag[all_local_ids])]
            tn_loc = f'/scratch/ydong/stamps/demo_F200W/F200W_{tn_id}.jpg'
            tn = np.array(Image.open(tn_loc))
            plt.imshow(tn,extent=(grid1[i],grid1[i+1],grid2[j],grid2[j+1]),cmap='gray')
            

plt.title('UMAP thumbnails')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.savefig('finetune_repr/repr_results/F200W_thumbnails_new.png')


time_end = time.time() 
print('>>> total time: %5.2f minutes'%((time_end - time_start)/60.))