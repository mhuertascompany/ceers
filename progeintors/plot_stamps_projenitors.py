import pandas as pd
import os
import numpy as np
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import pdb



def create_cutout(ra,dec,nir_f200_dict,rgb=['f356w','f200w','f115w']):
    read=0
    k=0
    while read==0:
        if k>=10:
            read=-1
            continue
        r=nir_f200_dict['images'][rgb[0]][k]
        g=nir_f200_dict['images'][rgb[1]][k]
        b=nir_f200_dict['images'][rgb[2]][k]
        wr=nir_f200_dict['wcs'][rgb[0]][k]
        wg=nir_f200_dict['wcs'][rgb[1]][k]
        wb=nir_f200_dict['wcs'][rgb[2]][k]
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
        return make_lupton_rgb(stamp_r, stamp_g, stamp_b)



def plot_color_stamps(nir_f200_dict,path_projenitors):
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

# Initialize an empty dictionary to store the images and headers
nir_f200_dict = {}
w = []

# List of wavelengths
wl_vec = ['f115w', 'f200w', 'f356w']

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



plot_color_stamps(nir_f200_dict,path_projenitors)

         
        #plot_stamps_quantiles(wl,morph,ceers_cat,nir_f200_list,w)