'''
Build from scratch the dataset by cutting out galaxy greyscale stamps. 
'''

from PIL import Image
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
import pandas as pd
import os
import numpy as np
from skimage.transform import resize

# convert a greyscale numpy array to [0,255] jpg image
def array2img(arr,clipped_percent=0):
    arr = np.arcsinh(arr)
    # max_val = np.percentile(arr,100-clipped_percent/2)
    # min_val = np.percentile(arr,clipped_percent/2)
    # arr = np.clip(arr,min_val,max_val)
    # arr = (arr-min_val)/(max_val-min_val)*255
    max = np.max(arr)
    min = np.min(arr)
    arr = (arr-min)/(max-min)*300.5-0.5
    arr = np.clip(arr,0.,255.)
    return Image.fromarray(arr.astype(np.uint8))

def zero_pix_fraction(img):
    zeros = np.sum(np.max(img,axis=0)==0.)+np.sum(np.max(img,axis=1)==0.)
    size = img.shape[0]
    return zeros/size

# directory for raw CEERS images
img_dir = "/scratch/ydong/images"

FILTER = 444

N_POINTINGS = 10
POINTING1 = [1,2,3,6]
POINTING2 = [4,5,7,8,9,10]

# path for catalog
cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

cat = pd.read_csv(os.path.join(cat_dir,cat_name))

col_names = ['RA_1','DEC_1']
coords = cat[col_names].values

fit_flag = cat[f'F{FILTER}W_FLAG'].values
star_flag = cat['star_flag'].values
Re = cat[f'F{FILTER}W_RE'].values
axis_ratio = cat[f'F{FILTER}W_Q'].values

len = len(fit_flag)

found = np.zeros(len)

for n in range(1,11):
    if n in POINTING1:
        img_name = "hlsp_ceers_jwst_nircam_nircam%i_f%iw_dr0.5_i2d.fits"%(n,FILTER)
    if n in POINTING2:
        img_name = "ceers_nircam%i_f%iw_v0.51_i2d.fits"%(n,FILTER)

    with fits.open(os.path.join(img_dir,img_name)) as hdul:
        # hdul.info()
        hdr = hdul[1].header
        w = wcs.WCS(hdr)
        data = hdul[1].data

        ymax, xmax = data.shape
        pixels = w.wcs_world2pix(coords,0)
        pix_size = 0.031

        for i in range(len):
            if (found[i]==0) & (fit_flag[i]==0) & (star_flag[i]==0):
                size = 212*np.maximum(0.04*Re[i]*np.sqrt(axis_ratio[i])/pix_size,0.1)
                pix = pixels[i]
                up = int(pix[0]+size)
                down = up-size*2
                right = int(pix[1]+size)
                left = right-size*2
                if all([up<xmax,down>-1,right<ymax,left>-1]):   
                    # cut = data[left:right,down:up]
                    cut = Cutout2D(data,pix,wcs=w,size=size*2).data

                    if zero_pix_fraction(cut)<0.1:  # exclude images with too many null pixels
                        print(i,n)
                        resized_cut = resize(cut,output_shape=(424,424))

                        image = array2img(resized_cut)

                        # save the images
                        image.save('/scratch/ydong/stamps/demo_F200W/F200W_%i.jpg'%i)

                        found[i] = 1

