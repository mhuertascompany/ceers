from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.python.keras import Model
import imageio
import pandas as pd
import numpy as np
import IPython.display as display
import os
import pickle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from official.vision.image_classification.augment import RandAugment

from sklearn.utils import shuffle
import pdb
from astropy.visualization import MinMaxInterval
interval = MinMaxInterval()
from astropy.visualization import AsinhStretch,LogStretch
from datetime import date

from tempfile import TemporaryFile


WRITE=True
TRAIN=False



import os

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def read_CANDELS_data_and_save(data_path, output_path):
    
    candels_cat = pd.read_csv(data_path + "QSOs/COSMOS_QSO_presample.csv")
    
    wfc3_f160_list = []
    wf160 = []
    candels_images = [
        "hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits",
        "hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits",
        "hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits",
        "hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits"
    ]
    

    for c in candels_images:
        wfc3_f160 = fits.open(data_path+"images/"+c)
        wfc3_f160_list.append(wfc3_f160)
        wf160.append(WCS(wfc3_f160[0].header))
        wf160[-1].sip = None

        
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fields = ["egs", "GDS", "COSMOS", "UDS"]

    for wfc3_f160, w_c, f in zip(wfc3_f160_list, wf160, fields): 
        for ra, dec in zip(candels_cat.RA, candels_cat.DEC):
            try:
                position = SkyCoord(ra, dec, unit="deg")
                stamp = Cutout2D(wfc3_f160[0].data, position, 32, wcs=w_c)
                
                if np.max(stamp.data) <= 0 or np.count_nonzero(stamp.data == 0) > 10:
                    continue

                #transform = AsinhStretch() + MinMaxInterval()
                #norm = transform(stamp.data)
                
                # Create a new FITS file for each stamp
                fits_filename = os.path.join(output_path, f"CANDELS_stamp_ra{ra}_dec{dec}.fits")
                hdu = fits.PrimaryHDU(stamp.data)
                hdul = fits.HDUList([hdu])
                hdul.writeto(fits_filename, overwrite=True)

            except:
                continue

# Example usage:
data_path = "/scratch/mhuertas/CEERS/data_release/"
output_path =  "/scratch/mhuertas/CEERS/data_release/QSOs/"
read_CANDELS_data_and_save(data_path, output_path)