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




def read_CANDELS_data(data_path):
    
    candels_cat = pd.read_csv(data_path+"QSOs/COSMOS_QSO_presample.csv")
    


    wfc3_f160_list=[]
    wf160=[]
    candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits"]
    for c in candels_images:
        wfc3_f160 = fits.open(data_path+"images/"+c)
        wfc3_f160_list.append(wfc3_f160)
        wf160.append(WCS(wfc3_f160[0].header))
        wf160[-1].sip = None



    fields = ["egs","GDS","COSMOS","UDS"]
    X=[]
    label=[]

    for wfc3_f160,w_c,f in zip(wfc3_f160_list,wf160,fields): 
        sel = candels_cat.query('HMAG<24 and FIELD=='+'"'+f+'"')
        print(len(sel))
        
        for idn,ra,dec,fsph,fdk,firr in zip(sel.RB_ID,sel.RA,sel.DEC,sel.F_SPHEROID,sel.F_DISK,sel.F_IRR):
                
                try:
                    
                    position = SkyCoord(ra,dec,unit="deg")
                    #print(ra,dec)
                    
                    stamp = Cutout2D(wfc3_f160[0].data,position,32,wcs=w_c)
                    
                    
                    
                    if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                        continue
                    
                    
                    transform = AsinhStretch() + interval
                    #transform = LogStretch() + interval
                    norm = transform(stamp.data)
                    
                    #stamp_name = data_path+"NirCam/CANDELS_stamps/v005/f200fullres/CANDELS-CEERS"+str(idn)+"_f200w_v005.fits"

                    if (fsph>=0.66 and fdk<0.66 and firr<0.1):
                        label.append(0)
                        
                        X.append(norm)
                    elif ((fsph<0.66 and fdk>=0.66 and firr<0.1)):
                        label.append(1)
                        X.append(norm)
                    elif ((firr>=0.1)):
                        label.append(2) 
                        X.append(norm)
                    elif ((fsph>0.66 and fdk>0.66 and firr<0.1)):
                        label.append(3)
                        X.append(norm)     


                except:
                    continue


    return X,label, candels_cat     


data_path = "/scratch/mhuertas/CEERS/data_release/"
X, label, candels_cat = read_CANDELS_data(data_path)

# Add 'label' as a new column to the 'candels_cat' DataFrame
candels_cat['label'] = label

# Save the 'X' data as an .npz file
np.savez(data_path+"QSOs/QSOs_CANDELS.npz", X=np.array(X))

# Save the updated 'candels_cat' DataFrame as a CSV file
candels_cat.to_csv(data_path+"QSOs/COSMOS_QSO_presample_morph.csv", index=False)