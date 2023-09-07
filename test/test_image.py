
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy import wcs
from astropy.nddata import Cutout2D
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

img_dir = "/scratch/ydong/images"
img_name = "ceers_nircam10_f150w_v0.51_i2d.fits"

cat_dir = "/scratch/mhuertas/CEERS/data_release/cats"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v051_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))
col_names = ['RA_1','DEC_1']

coords = cat[col_names].values

with fits.open(os.path.join(img_dir,img_name)) as hdul:
    print(repr(hdul[0].header))
    hdr = hdul[1].header
    w = wcs.WCS(hdr)
    print(repr(hdr))
    data = hdul[1].data

# zero = np.sum(data==0.)
# plt.imshow(data==0.,cmap='gray')
# plt.savefig('zero7.png')

# fig = plt.figure(figsize=(10, 10))
# ax = plt.subplot(projection=w)
# norm = simple_norm(data,'log',percent=98)
# plt.imshow(data, origin='lower', cmap='cividis', aspect='equal')
# plt.xlabel(r'Ra')
# plt.ylabel(r'Dec')

# overlay = ax.get_coords_overlay('icrs')
# overlay.grid(color='white', ls='dotted')

# plt.savefig('test8.png')

size = 100
# print(w)
print(data.shape)
ymax, xmax = data.shape
pixels = w.wcs_world2pix(coords,0)
for i in range(24320):
    pix = pixels[i]
    up = int(pix[0]+size)
    down = up-size*2
    right = int(pix[1]+size)
    left = right-size*2
    if all([up<xmax,down>-1,right<ymax,left>-1]):
        
        # cut = data[left:right,down:up]
        cut = Cutout2D(data,pix,wcs=w,size=size*2)
        if np.max(cut.data)>0:
            print(i)
            print(pix,coords[i],w.wcs_pix2world(pix.reshape(-1,2),0))
            norm = simple_norm(cut.data,'log',percent=99)
            
            plt.imshow(cut.data,cmap='gray')
            
            plt.savefig('sw10_%i.png'%i)

