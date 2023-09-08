from PIL import Image
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy import wcs
from astropy.nddata import Cutout2D
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# convert a greyscale numpy array to [0,255] jpg image
def array2img(arr,clipped_percentile=0):
    max_val = np.percentile(arr,100-clipped_percentile/2)
    min_val = np.percentile(arr,clipped_percentile/2)
    arr = np.clip(arr,min_val,max_val)
    arr = (arr-min_val)/(max_val-min_val)*255
    return Image.fromarray(arr.astype(np.uint8))

img_dir = "/scratch/ydong/images"
img_name = "ceers_nircam10_f200w_v0.51_i2d.fits"

cat_dir = "/scratch/mhuertas/CEERS/data_release/cats"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v051_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))
col_names = ['RA_1','DEC_1']

coords = cat[col_names].values

with fits.open(os.path.join(img_dir,img_name)) as hdul:
    hdul.info()
    hdr = hdul[1].header
    w = wcs.WCS(hdr)
    # print(repr(hdr))
    data = hdul[1].data
    con = hdul[3].data

plt.imshow(con[0],cmap='gray')
plt.savefig('con10.png')

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
print(data.shape)
ymax, xmax = data.shape
pixels = w.wcs_world2pix(coords,0)
for i in range(11000):
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
            # norm = simple_norm(cut.data,'log',percent=99)
            
            # plt.imshow(cut.data,cmap='gray')
            
            # plt.savefig('sw10_%i.png'%i)
            image = array2img(cut.data)

            image.save('images/demo_train/sw_%i.jpg'%i)

