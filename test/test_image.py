
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy import wcs
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

img_dir = "/scratch/ydong/images"
img_name = "ceers_nircam10_f115w_v0.51_i2d.fits"

cat_dir = "/scratch/mhuertas/CEERS/data_release/cats"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v051_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))
col_names = ['RA_1','DEC_1']

coords = cat[col_names].values

with fits.open(os.path.join(img_dir,img_name)) as hdul:
    # hdul.info()
    hdr = hdul[1].header
    w = wcs.WCS(hdr)
    data = hdul[1].data

zero = np.sum(data==0.)
plt.imshow(data==0.,cmap='gray')
plt.savefig('zero.png')

norm = simple_norm(data,'log',percent=99)
plt.imshow(data,norm=norm,cmap='gray')
plt.savefig('test.png')

size = 100
# print(w)
pixels = w.wcs_world2pix(coords,0).astype(int)
for i in range(10000):
    pix = pixels[i]
    up = pix[0]+size
    down = up-size*2
    right = pix[1]+size
    left = right-size*2
    if all([up<4800,down>-1,right<10500,left>-1]):
        print(i)
        cut = data[down:up,left:right]
        norm = simple_norm(cut,'log',percent=99)
        plt.imshow(cut,norm=norm)
        plt.savefig('test_%i.png'%i)

