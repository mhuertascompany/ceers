import pandas as pd
import os
import numpy as np
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import pdb
from astropy.io import fits
import matplotlib.pyplot as plt


def create_cutout(ra,dec,nir_f200_dict,rgb=['f356w','f200w','f150w']):
    read=0
    k=0
    while read==0:
        if k>=10:
            read=-1
            continue
        r = nir_f200_dict[rgb[0]]['images'][k]
        g = nir_f200_dict[rgb[1]]['images'][k]  # Corrected line
        b = nir_f200_dict[rgb[2]]['images'][k]  # Corrected line
        wr = nir_f200_dict[rgb[0]]['wcs'][k]     # Corrected line
        wg = nir_f200_dict[rgb[1]]['wcs'][k]     # Corrected line
        wb = nir_f200_dict[rgb[2]]['wcs'][k]     # Corrected line
        k+=1
        try:
            position = SkyCoord(ra,dec,unit="deg")
            stamp_r = Cutout2D(r[1].data,position,64,wcs=wr)
            stamp_g = Cutout2D(g[1].data,position,64,wcs=wg)
            stamp_b = Cutout2D(b[1].data,position,64,wcs=wb)
            if np.max(stamp_r.data)<=0 or np.count_nonzero(stamp_r.data==0)>10:
                continue
            print("read!")
            read=1
        except:
            print("error reading")
            continue    
    if read == -1:
        return make_lupton(np.zeros((64,64)),np.zeros((64,64)),np.zeros((64,64))) 
    else:
        return make_lupton_rgb(stamp_r.data, stamp_g.data, stamp_b.data)



def plot_color_stamps(nir_f200_dict,path_projenitors,output_dir):
    for filename in os.listdir(path_projenitors):
        if filename.endswith(".csv") and filename.startswith('CEERS'):
            proj = pd.read_csv(os.path.join(path_projenitors, filename))
            ra_proj=proj.ra
            dec_proj=proj.dec
            num_images = len(ra_proj)

            # Create a figure with subplots for each RGB image
            fig, axs = plt.subplots(1, num_images, figsize=(num_images * 8, 8))

            for i, (ra, dec) in enumerate(zip(ra_proj, dec_proj)):
                rgb_image=create_cutout(ra,dec,nir_f200_dict)
                axs[i].imshow(rgb_image)
                axs[i].axis('off')

            # Define the output image file name
            output_filename = os.path.join(output_dir, filename.replace(".csv", "_rgb.png"))
            # Save the combined image to the output file
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close()


data_path = "/scratch/mhuertas/CEERS/data_release/"
path_projenitors = '/scratch/mhuertas/CEERS/proj/projenitors'
output_dir = '/scratch/mhuertas/CEERS/proj/projenitors/rgb/'

# Initialize an empty dictionary to store the images and headers
nir_f200_dict = {}
w = []

# List of wavelengths
wl_vec = ['f150w', 'f200w', 'f356w']

# Iterate through the wavelengths
for wl in wl_vec:
    ceers_pointings = np.arange(1, 11)
    nir_f200_list = []
    wcs_list = []

    for c in ceers_pointings:
        if c == 1 or c == 2 or c == 3 or c == 6:
            nir_f200 = fits.open(data_path + "images/hlsp_ceers_jwst_nircam_nircam" + str(c) + "_" + wl + "_dr0.5_i2d.fits.gz")
        else:
            nir_f200 = fits.open(data_path + "images/ceers_nircam" + str(c) + "_" + wl + "_v0.51_i2d.fits.gz")

        nir_f200_list.append(nir_f200)
        wcs_list.append(WCS(nir_f200[1].header))

    # Store the list of images, headers, and WCS in the dictionary
    nir_f200_dict[wl] = {
        'images': nir_f200_list,
        'wcs': wcs_list
    }



plot_color_stamps(nir_f200_dict,path_projenitors,otuput_dir)

         
        #plot_stamps_quantiles(wl,morph,ceers_cat,nir_f200_list,w)