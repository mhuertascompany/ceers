from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.python.keras import Model
#import imageio
import pandas as pd
import numpy as np
import IPython.display as display
import os
#import pickle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import astropy.wcs as wcs
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
#from official.vision.image_classification.augment import RandAugment

from sklearn.utils import shuffle
import pdb
from astropy.visualization import MinMaxInterval
interval = MinMaxInterval()
from astropy.visualization import AsinhStretch,LogStretch
from datetime import date

from tempfile import TemporaryFile

import gc

import random


WRITE=True
TRAIN=True
WRITE_CANDELS=True



import os

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


filters = [
    'F115W', 'F150W',  'F277W', 'F444W', 
    'F770W', 
    'HST-F814W', 
    'CFHT-u', 
    'HSC-g', 'HSC-r', 'HSC-i', 'HSC-z', 'HSC-y', 
    'HSC-NB0816', 'HSC-NB0921', 'HSC-NB1010',
    'UVISTA-Y', 'UVISTA-J', 'UVISTA-H', 'UVISTA-Ks', 'UVISTA-NB118', 
    'SC-IA484', 'SC-IA527', 'SC-IA624', 'SC-IA679', 'SC-IA738', 'SC-IA767', 'SC-IB427', 
    ####'SC-IB464', 
    'SC-IB505', 'SC-IB574', 'SC-IB709', 'SC-IB827', 'SC-NB711', 'SC-NB816'
]

filters_FF = [
    'F770W', 
    'CFHT-u', 
    'HSC-g', 'HSC-r', 'HSC-i', 'HSC-z', 'HSC-y', 
    'HSC-NB0816', 'HSC-NB0921', 'HSC-NB1010',
    'UVISTA-Y', 'UVISTA-J', 'UVISTA-H', 'UVISTA-Ks', 'UVISTA-NB118', 
    'SC-IA484', 'SC-IA527', 'SC-IA624', 'SC-IA679', 'SC-IA738', 'SC-IA767', 'SC-IB427', 
    ####'SC-IB464', 
    'SC-IB505', 'SC-IB574', 'SC-IB709', 'SC-IB827', 'SC-NB711', 'SC-NB816'
]

filters_translate = {
    'F115W':       f'F115W',
    'F150W':       f'F150W',
    'F277W':       f'F277W',
    'F444W':       f'F444W',
    'F770W':       f'F770W',
    'HST-F814W':   f'f814w',
    'CFHT-u':      f'COSMOS.U2',
    'HSC-g':       f'HSC-G',
    'HSC-r':       f'HSC-R',
    'HSC-i':       f'HSC-I',
    'HSC-z':       f'HSC-Z',
    'HSC-y':       f'HSC-Y',
    'HSC-NB0816':  f'NB0816',
    'HSC-NB0921':  f'NB0921',
    'HSC-NB1010':  f'NB1010',
    'UVISTA-Y':    f'UVISTA_Y',
    'UVISTA-J':    f'UVISTA_J',
    'UVISTA-H':    f'UVISTA_H',
    'UVISTA-Ks':   f'UVISTA_Ks',
    'UVISTA-NB118':f'UVISTA_NB118',
    'SC-IA484':    f'SPC_L484',
    'SC-IA527':    f'SPC_L527',
    'SC-IA624':    f'SPC_L624',
    'SC-IA679':    f'SPC_L679',
    'SC-IA738':    f'SPC_L738',
    'SC-IA767':    f'SPC_L767',
    'SC-IB427':    f'SPC_L427',
    'SC-IB505':    f'SPC_L505',
    'SC-IB574':    f'SPC_L574',
    'SC-IB709':    f'SPC_L709',
    'SC-IB827':    f'SPC_L827',
    'SC-NB711':    f'SPC_L711',
    'SC-NB816':    f'SPC_L816'
    }

band_lambda_micron = {
                'CFHT-u'        : 0.386,
                'SC-IB427'      : 0.426,
                'HSC-g'         : 0.475,
                'SC-IA484'      : 0.485,
                'SC-IB505'      : 0.506,
                'SC-IA527'      : 0.526,
                'SC-IB574'      : 0.576,
                'HSC-r'         : 0.623,
                'SC-IA624'      : 0.623,
                'SC-IA679'      : 0.678,
                'SC-IB709'      : 0.707,
                'SC-NB711'      : 0.712,
                'SC-IA738'      : 0.736,
                'SC-IA767'      : 0.769,
                'HSC-i'         : 0.770,
                'SC-NB816'      : 0.815,
                'HSC-NB0816'    : 0.816,
                'HST-F814W'     : 0.820,
                'SC-IB827'      : 0.824,
                'HSC-z'         : 0.890,
                'HSC-NB0921'    : 0.921,
                'HSC-y'         : 1.000,
                'HSC-NB1010'    : 1.010,
                'UVISTA-Y'      : 1.102,
                'F115W'         : 1.150,      
                'UVISTA-NB118'  : 1.191,
                'UVISTA-J'      : 1.252,
                'F150W'         : 1.501,
                'UVISTA-H'      : 1.647,
                'UVISTA-Ks'     : 2.156,
                'F277W'         : 2.760, 
                'F444W'         : 4.408, 
                'F770W'         : 7.646
} 



def read_CANDELS_data(data_path):
    
    candels_cat = pd.read_csv(data_path+"cats/CANDELS_morphology.csv")
    


    wfc3_f160_list=[]
    wf160=[]
    candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gn-tot-60mas_f160w_v1.0_drz.fits"]
    for c in candels_images:
        wfc3_f160 = fits.open(data_path+"images/"+c)
        wfc3_f160_list.append(wfc3_f160)
        wf160.append(WCS(wfc3_f160[0].header))
        wf160[-1].sip = None



    fields = ["egs","GDS","COSMOS","UDS","gdn"]
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


    return X,label                     

def read_CEERS_data(filter,data_path):
    cat_ceers =   pd.read_csv(data_path+"CEERS_v0.51_photom.csv")
    cat_ceers["F200_AB"] = 2.5*(23-np.log10(cat_ceers.FLUX_200*1e-9))-48.6

    ceers_pointings = np.arange(1,11) #["1","2","3","4","5","6","7","8","9","10"]
    nir_f200_list=[]
    w=[]
    cats = []
    for c in ceers_pointings:
        if c==1 or c==2 or c==3 or c==6:
            nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+str(c)+"_"+filter+"_dr0.5_i2d.fits.gz")
        else:
            nir_f200 = fits.open(data_path+"images/ceers_nircam"+str(c)+"_"+filter+"_v0.51_i2d.fits.gz")
            # ceers_nircam10_f356w_v0.51_i2d.fits.gz    
        nir_f200_list.append(nir_f200)
        w.append(WCS(nir_f200[1].header))
        cats.append(cat_ceers.query("FIELD=="+str(c)))

    X_JWST=[]
    idvec=[]
    fullvec=[]
    fieldvec=[]
    ravec=[]
    decvec=[]
    for nir_f200,w_v,cat in zip(nir_f200_list,w,cats):
    
        sel = cat.query('F200_AB<27 and F200_AB>0')
        #print(cat)  
        for idn, field, ra,dec in zip(sel.CATID, sel.FIELD, sel.RA,sel.DEC):
                try:
                    full = 'nircam_'+str(field)+'_'+str(idn)
                    position = SkyCoord(ra,dec,unit="deg")
                    #print(ra,dec)
                    stamp = Cutout2D(nir_f200['SCI'].data,position,32,wcs=w_v)
                    
                    if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                        continue
                    
                    transform = AsinhStretch() + interval
                    norm = transform(stamp.data)  
                    #pdb.set_trace()
                    #stamp_name = data_path+"NirCam/CANDELS_stamps/v005/f200fullres/CANDELS-CEERS"+str(idn)+"_f200w_v005.fits"
                    X_JWST.append(norm)
                    idvec.append(idn)
                    fullvec.append(full)
                    fieldvec.append(field) 
                    ravec.append(ra)
                    decvec.append(dec)  
                    #if (fsph>0.66 and fdk<0.66 and firr<0.1):
                    #    label.append(1)
                    #else:
                    #    label.append(0)
                
                    
                except:
                    continue

    return X_JWST,fullvec,idvec,fieldvec,ravec,decvec                   


def get_filename(path, filt, key):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if filt in file:
                if key in file:
                    return str(file)

def load_imgs(tile,bands_to_plot):

    if tile in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        name_img_det        = f'/n17data/lpaquereau/COSMOSWeb/JAN24/mosaics/mosaic_nircam_chimean-SWLW_COSMOS-Web_30mas_{tile}.fits'
        sci_imas={
            'F115W':       f'/n17data/shuntov/COSMOS-Web/Images_JAN24/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_sci.fits',
            'F150W':       f'/n17data/shuntov/COSMOS-Web/Images_JAN24/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_sci.fits',
            'F277W':       f'/n17data/shuntov/COSMOS-Web/Images_JAN24/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_sci.fits',
            'F444W':       f'/n17data/shuntov/COSMOS-Web/Images_JAN24/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_sci.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/JAN24_v0.6/mosaic_miri_f770w_COSMOS-Web_30mas_{tile}_v0_6_sci.fits',
            'HST-F814W':   f'/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles/mosaic_cosmos_web_2024jan_30mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits',
            'CFHT-u':      f'/n17data/shuntov/CWEB-GroundData-Tiles/COSMOS.U2.clipped_zp-28.09_{tile}.fits',
            'HSC-g':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-G-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-r':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-R-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-i':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-I-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-z':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-Z-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-y':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-Y-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB0816':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB0816-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB0921':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB0921-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB1010':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB1010-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'SC-IA484':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L484_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA527':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L527_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA624':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L624_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA679':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L679_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA738':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L738_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA767':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L767_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB427':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L427_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB505':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L505_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB574':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L574_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB709':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L709_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB827':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L827_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB711':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L711_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits'
        }

    if tile == 'JAN':
        name_img_det        = f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-chi2-SWLW_COSMOS-Web_30mas_resc100.fits'
        sci_imas={
            'F115W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F115W_30mas_sci.fits',
            'F150W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F150W_30mas_sci.fits',
            'F277W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F277W_30mas_sci.fits',
            'F444W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F444W_30mas_sci.fits',
            'F770W':      f'/n17data/shuntov/COSMOS-Web/Images_MIRI/v0.11/mosaic-MIRI_v0.2_F770W_60mas_sci_zp-28.09.fits',
            'HST-F814W':  f'/n17data/shuntov/COSMOS-Web/Images_HST-ACS/mosaic_cosmos_web_2023jan_30mas_hst_acs_wfc_f814w_drz.fits',
            'CFHT-u':      f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_COSMOS.U2.clipped_zp-28.09.fits',
            'HSC-g':       f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-HSC-G-9813-pdr3_dud_rev-230405-092448_sci_fscaled.fits',
            'HSC-r':       f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-HSC-R-9813-pdr3_dud_rev-230405-092451_sci_fscaled.fits',
            'HSC-i':       f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-HSC-I-9813-pdr3_dud_rev-230405-092454_sci_fscaled.fits',
            'HSC-z':       f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-HSC-Z-9813-pdr3_dud_rev-230405-092457_sci_fscaled.fits',
            'HSC-y':       f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-HSC-Y-9813-pdr3_dud_rev-230405-092500_sci_fscaled.fits',
            'HSC-NB0816':  f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-NB0816-9813-pdr3_dud_rev-230405-092516_sci_fscaled.fits',
            'HSC-NB0921':  f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-NB0921-9813-pdr3_dud_rev-230405-092553_sci_fscaled.fits',
            'HSC-NB1010':  f'/n17data/shuntov/COSMOS-Web/HSC-PDR3/cutout-NB1010-9813-pdr3_dud_rev-230405-092607_sci_fscaled.fits',
            'UVISTA-Y':    f'/n07data/shuntov/UVISTA_DR5/cutouts/CUTOUT-Full-JAN23_UVISTA_Y_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09.fits',
            'UVISTA-J':    f'/n07data/shuntov/UVISTA_DR5/cutouts/CUTOUT-Full-JAN23_UVISTA_J_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09.fits',
            'UVISTA-H':    f'/n07data/shuntov/UVISTA_DR5/cutouts/CUTOUT-Full-JAN23_UVISTA_H_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09.fits',
            'UVISTA-Ks':   f'/n07data/shuntov/UVISTA_DR5/cutouts/CUTOUT-Full-JAN23_UVISTA_Ks_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09.fits',
            'UVISTA-NB118':f'/n07data/shuntov/UVISTA_DR5/cutouts/CUTOUT-Full-JAN23_UVISTA_NB118_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09.fits',
            'SC-IA484':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L484_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IA527':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L527_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IA624':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L624_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IA679':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L679_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IA738':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L738_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IA767':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L767_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IB427':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L427_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IB505':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L505_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IB574':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L574_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IB709':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L709_20-09-29a_cosmos_zp-28.09.fits',
            'SC-IB827':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L827_20-09-29a_cosmos_zp-28.09.fits',
            'SC-NB711':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L711_20-09-29a_cosmos_zp-28.09.fits',
            'SC-NB816':    f'/n07data/shuntov/COSMOS2020_Images/cutouts/CUTOUT-Full-JAN23_SPC_L816_20-09-29a_cosmos_zp-28.09.fits'
            }

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        name_img_det        = f'/n17data/shuntov/COSMOS-Web/Images_April/mosaic-chi2-SWLW_COSMOS-Web_30mas_{tile}.fits'
        sci_imas={
            'F115W':       f'/n23data2/lpaquereau/COSMOSWeb/APR23/mosaics_v0.002/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_002_i2d_split.fits',
            'F150W':       f'/n23data2/lpaquereau/COSMOSWeb/APR23/mosaics_v0.002/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_002_i2d_split.fits',
            'F277W':       f'/n23data2/lpaquereau/COSMOSWeb/APR23/mosaics_v0.002/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_002_i2d_split.fits',
            'F444W':       f'/n23data2/lpaquereau/COSMOSWeb/APR23/mosaics_v0.002/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_002_i2d_split.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/APR23_v0.001/mosaic_miri_f770w_COSMOS-Web_30mas_{tile}_v0_01_sci.fits',
            'HST-F814W':   f'/n17data/shuntov/COSMOS-Web/Images_HST-ACS/AprilTiles/mosaic_cosmos_web_2023apr_30mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits',
            'CFHT-u':      f'/n17data/shuntov/CWEB-GroundData-Tiles/COSMOS.U2.clipped_zp-28.09_{tile}.fits',
            'HSC-g':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230412-135737_sci_zp-28.09_{tile}.fits',
            'HSC-r':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-121613_sci_zp-28.09_{tile}.fits',
            'HSC-i':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-121625_sci_zp-28.09_{tile}.fits',
            'HSC-z':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-121629_sci_zp-28.09_{tile}.fits',
            'HSC-y':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-121631_sci_zp-28.09_{tile}.fits',
            'HSC-NB0816':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-121622_sci_zp-28.09_{tile}.fits',
            'HSC-NB0921':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-121626_sci_zp-28.09_{tile}.fits',
            'HSC-NB1010':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-121845_sci_zp-28.09_{tile}.fits', 
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_07_22_allpaw_skysub_015_dr5_rc_v1_zp-28.09_{tile}.fits',
            'SC-IA484':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L484_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA527':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L527_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA624':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L624_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA679':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L679_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA738':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L738_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA767':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L767_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB427':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L427_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB505':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L505_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB574':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L574_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB709':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L709_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB827':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L827_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB711':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L711_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits'
            }
        if tile in ['A1', 'A2', 'A3', 'A8', 'A7', 'A6']:
            sci_imas['HSC-g'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230412-135737_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-r'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-121613_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-i'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-121625_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-z'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-121629_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-y'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-121631_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0816'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-121622_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0921'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-121626_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB1010'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-121845_sci_zp-28.09_{tile}.fits'

        elif tile in ['A4', 'A5', 'A9', 'A10']:
            sci_imas['HSC-g'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-r'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-130346_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-i'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-130351_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-z'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-130355_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-y'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0816'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0921'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-130520_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB1010'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-130522_sci_zp-28.09_{tile}.fits'

    filters_translate['F115W'] = 'f115w'
    filters_translate['F150W'] = 'f150w'
    filters_translate['F277W'] = 'f277w'
    filters_translate['F444W'] = 'f444w'
    filters_translate['F770W'] = 'f770w'



    imgname_chi2_c20 = '/n08data/COSMOS2020/images/COSMOS2020_izYJHKs_chimean-v3.fits'

    if tile in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        ver = 'v2.1.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/JAN24-{tile}_{ver}-FF/'

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        ver = 'v1.5.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/APR-{tile}_{ver}-FF/'

    name_img_part  = path_checkimg+get_filename(path_checkimg, '', '_partition.fits')
    model_imas = {}
    resid_imas = {}

    # print(path_checkimg)
    for filt in bands_to_plot:
        f = get_filename(path_checkimg, filters_translate[filt], 'model')
        try:
            model_imas[filt] = path_checkimg + f
        except TypeError as err:
            print('Error for', tile, ':', err, path_checkimg, f)

        f = get_filename(path_checkimg, filters_translate[filt], 'resid')
        try:
            resid_imas[filt] = path_checkimg + f
        except TypeError as err:
            print('Error for', tile, ':', err, 'and filter', filt, path_checkimg, f)
            print('\n There seems to be an error for this galaxy\n')


    return name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate
        
            
    


    return name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate
    
def image_make_cutout(filename, ra1, dec1, arcsec_cut, nameout=None, get_wcs=None):
    import os
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy import units as u
    from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

    if 'miri' in filename:
        try:
            image_data = fits.getdata(filename, ext=1)
            hdu = fits.open(filename)[1]
            # get image WCS
            w = WCS(hdu.header)
        except:
            image_data = fits.getdata(filename, ext=0)
            hdu = fits.open(filename)[0]
            w = WCS(filename)  
    else:
        image_data = fits.getdata(filename, ext=0)
        hdu = fits.open(filename)[0]
        w = WCS(filename)
    

    # get the pixels from the defined sky coordinates
    sc_1 = SkyCoord(ra1, dec1, unit='deg')
#     sc_2 = SkyCoord(ra2, dec2, unit='deg')
    sc_pix_1 = skycoord_to_pixel(sc_1, w)
#     sc_pix_2 = skycoord_to_pixel(sc_2, w)
#     size_pix_ra = np.int(np.abs(sc_pix_1[0]-sc_pix_2[0]))
#     size_pix_dec = np.int(np.abs(sc_pix_1[1]-sc_pix_2[1]))

    ny, nx = image_data.shape
    image_pixel_scale = wcs.utils.proj_plane_pixel_scales(w)[0]
    image_pixel_scale *= w.wcs.cunit[0].to('arcsec')

    size_pix_ra = arcsec_cut/image_pixel_scale
    size_pix_dec = arcsec_cut/image_pixel_scale

        
#     print(sc_pix_1)
    pos_pix_ra = int((sc_pix_1[0])) #+sc_pix_2[0])/2)
    pos_pix_dec = int((sc_pix_1[1])) #+sc_pix_2[1])/2)

    ## perform the cut
#     try:
    cutout = Cutout2D(image_data, (pos_pix_ra, pos_pix_dec), (size_pix_dec, size_pix_ra), wcs=w)
    
    # Put the cutout image in the FITS HDU
    datacutout = cutout.data
    
    # Update the FITS header with the cutout WCS
#     hdu = fits.PrimaryHDU(data=datacutout, header=cutout.wcs.to_header())
#     hdu.header.update(cutout.wcs.to_header())
    
    # Write the cutout to a new FITS file
#     cutout_filename = '/n08data/shuntov/Images/'+nameout
#     cutout_filename = nameout
#     hdu.writeto(cutout_filename, overwrite=True)
    if get_wcs == True:
        return datacutout, cutout.wcs
    else:
        return datacutout    


def read_COSMOS_data(f,COSMOS_path):
    #name_SEpp_cat = COSMOS_path+"cats/COSMOSWeb_master_v1.6.0-sersic+BD-em_cgs_LePhare_nodupl_nomulti.fits"
    name_SEpp_cat = COSMOS_path+"cats/COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM.fits"
    #/n17data/shuntov/COSMOS-Web/Catalogs/COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM.fits
    #ith fits.open(name_SEpp_cat) as hdu:
    cat_cosmos = Table.read(name_SEpp_cat, format='fits')
    #cat_cosmos = hdu[1].data
    names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
    
    cat_cosmos_pd=cat_cosmos[names].to_pandas()

    sel = cat_cosmos_pd.query("MAG_MODEL_F150W<27 and MAG_MODEL_F150W>0 and TILE !='JAN'")
    
    source_ids = sel['Id']
    tiles = sel['TILE']
    ra  = sel['RA_MODEL']
    dec = sel['DEC_MODEL']
    arcsec_cut = 32*0.03
    X_JWST=[]
    idvec=[]
    fullvec=[]
    fieldvec=[]
    ravec=[]
    decvec=[]

    for idn,t,ra_cent,dec_cent in zip(source_ids,tiles,ra,dec):
        name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate = load_imgs(t.decode('utf-8'),f)
        
        print(name_img_det)
        #try:
        stamp, w = image_make_cutout(name_img_det, ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=True)
        print(stamp.shape)
        #except:
        #    print('Error creating stamp')
        full = 'nircam_'+str(t.decode())+'_'+str(idn)
        if stamp.shape[0] !=32 or stamp.shape[1] !=32 or np.max(stamp)<=0 or np.count_nonzero(stamp==0)>10:
            print("skipping")
            continue
                    
        transform = AsinhStretch() + interval
        norm = transform(stamp)  
        
        #pdb.set_trace()
        
        #stamp_name = data_path+"NirCam/CANDELS_stamps/v005/f200fullres/CANDELS-CEERS"+str(idn)+"_f200w_v005.fits"
        X_JWST.append(norm)
        idvec.append(idn)
        fullvec.append(full)
        fieldvec.append(t.decode()) 
        ravec.append(ra_cent)
        decvec.append(dec_cent)  

    return X_JWST,fullvec,idvec,fieldvec,ravec,decvec 

def create_datasets(X_C,label_C,X_JWST,sh=True,n_JWST=25000):

    train_s=len(X_C)*4//5
    test_s=len(X_C)*1//5
    if sh==True:
        print("I am shuffling the training")
        X, label = shuffle(X_C, label_C, random_state=0)
    else:
        X=X_C
        label=label_C    

  


    print(np.array(X).shape)
    with tf.device('CPU'):
        CANDELS_X = tf.convert_to_tensor(X[0:train_s], dtype=tf.float32)
        CANDELS_X = tf.expand_dims(CANDELS_X, -1)
        #CANDELS_X = tf.tile(CANDELS_X, [1,1,1,3])
        print(tf.shape(CANDELS_X))
        label_candels = tf.one_hot(label[0:train_s], 4).numpy()

        CANDELS_X_t = tf.convert_to_tensor(X[train_s:train_s+test_s], dtype=tf.float32)
        CANDELS_X_t = tf.expand_dims(CANDELS_X_t, -1)
        #CANDELS_X = tf.tile(CANDELS_X, [1,1,1,3])
        print(tf.shape(CANDELS_X_t))
        label_candels_t = tf.one_hot(label[train_s:train_s+test_s], 4).numpy()
        print(np.array(X_JWST).shape)
        #X_JWST_sampled = random.sample(X_JWST, n_JWST)
        indices = np.random.choice(np.array(X_JWST).shape[0], n_JWST, replace=False)
        X_JWST_sampled = (np.array(X_JWST))[indices.astype(int)]
        print(np.array(X_JWST_sampled).shape)

        JWST_X_sampled = tf.convert_to_tensor(X_JWST_sampled, dtype=tf.float32)
        JWST_X_sampled = tf.expand_dims(JWST_X_sampled, -1)
        JWST_X = tf.convert_to_tensor(X_JWST, dtype=tf.float32)
        JWST_X = tf.expand_dims(JWST_X, -1)

        label_JWST = np.zeros(len(JWST_X_sampled))
        #JWST_X = tf.tile(JWST_X, [1,1,1,3])
        label_JWST = tf.one_hot(np.zeros(len(JWST_X_sampled)), 4).numpy()

    return CANDELS_X,label_candels,CANDELS_X_t,label_candels_t,JWST_X_sampled,label_JWST,JWST_X




def get_network(image_size=32, num_classes=4):
  inputs = keras.Input(shape=(image_size, image_size, 1))
  rot = tf.keras.layers.RandomRotation(0.25)(inputs)
  flip = tf.keras.layers.RandomFlip()(rot)
  conv1 = layers.Conv2D(
        16,
        (2, 2),
        strides=1,
        padding="same",
        activation = "relu"
    )(flip)
  bn1 = layers.BatchNormalization()(conv1)
  conv2 =  layers.Conv2D(
        64,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn1)
  mp1 = layers.MaxPool2D((2,2))(conv2)
  bn2 =  layers.BatchNormalization()(mp1)
  conv3 =  layers.Conv2D(
        128,
        (3, 3),
        strides=1,
        padding="same",
        activation = "relu"
    )(bn2)
  mp2 = layers.MaxPool2D((2,2))(conv3)
  bn3 = layers.BatchNormalization()(mp2)
  conv4 =  layers.Conv2D(
        128,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn3)
  bn4 = layers.BatchNormalization()(conv4)
  conv5 =  layers.Conv2D(
        128,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn4)
  bn5 = layers.BatchNormalization()(conv5)
  #trunk_outputs = layers.GlobalAveragePooling2D()(bn5)
  outputs = layers.Flatten()(bn5) 
  #d1 = layers.Dense(64, activation = "relu")(fl)
  outputs = layers.Dropout(0.4)(outputs)
  #outputs = layers.Dense(num_classes)(dr1)
  return keras.Model(inputs, outputs)


class LabelPredictor(Model):
  def __init__(self):
    super(LabelPredictor, self).__init__() 
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(4, activation='softmax')
    self.dr = layers.Dropout(0.4)

  def call(self, feats):  
    feats = self.d1(feats)
    feats = self.dr(feats)
    return self.d2(feats)    


class DomainPredictor(Model):
  def __init__(self):
    super(DomainPredictor, self).__init__()   
    self.d3 = Dense(128, activation='relu')
    self.d4 = Dense(64, activation='relu')
    self.d5 = Dense(2, activation='softmax')
    self.dr = layers.Dropout(0.4)

  def call(self, feats):
    feats = self.d3(feats)
    feats = self.d4(feats)
    feats = self.dr(feats)
    return self.d5(feats)

    
@tf.function
def train_step(images, labels, images2, domains,alpha):
    
  """
  i. images = batch of source images
  ii. labels = corresponding labels
  iii. images2 = batch of source and target images
  iv. domains = corresponding domain labels
  v. alpha = weight attributed to the domain loss
  """
    
  ## Update the generator and the classifier
   
  with tf.GradientTape(persistent=True) as tape:
     
    features = feature_generator(images)
    l_predictions = label_predictor(features)
    #pdb.set_trace()
    #print(l_predictions.eval())
    features = feature_generator(images2)
    #pdb.set_trace()
    d_predictions = domain_predictor(features)
    label_loss = loss_object(labels, l_predictions)
    domain_loss = loss_object(domains, d_predictions)
    
  f_gradients_on_label_loss = tape.gradient(label_loss, feature_generator.trainable_variables)
  f_gradients_on_domain_loss = tape.gradient(domain_loss, feature_generator.trainable_variables)    
  f_gradients = [f_gradients_on_label_loss[i] - alpha*f_gradients_on_domain_loss[
      i] for i in range(len(f_gradients_on_domain_loss))]

    
  l_gradients = tape.gradient(label_loss, label_predictor.trainable_variables)

  f_optimizer.apply_gradients(zip(f_gradients+l_gradients, 
                                  feature_generator.trainable_variables+label_predictor.trainable_variables)) 
    
    
  ## Update the discriminator: Comment this bit to complete all updates in one step. Asynchronous updating 
  ## seems to work a bit better, with better accuracy and stability, but may take longer to train    
  with tf.GradientTape() as tape:
    features = feature_generator(images2)
    d_predictions = domain_predictor(features)
    #print(d_predictions)
    domain_loss = loss_object(domains,d_predictions)
  #####
   
  d_gradients = tape.gradient(domain_loss, domain_predictor.trainable_variables)  
  d_gradients = [alpha*i for i in d_gradients]
  d_optimizer.apply_gradients(zip(d_gradients, domain_predictor.trainable_variables))
  
    
  train_loss(label_loss)
  #print(label_loss)  
  train_accuracy(labels, l_predictions)
  conf_train_loss(domain_loss)
  conf_train_accuracy(domains, d_predictions)
  #print("TEST:", tf.print(domains))
  #pdb.set_trace()


@tf.function
def test_step(mnist_images, labels, mnist_m_images, labels2):
  #pdb.set_trace()  
  features = feature_generator(mnist_images)
  predictions = label_predictor(features)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

  features = feature_generator(mnist_m_images)
  predictions = label_predictor(features)
  t_loss = loss_object(labels2, predictions)
    
  m_test_loss(t_loss)
  m_test_accuracy(labels2, predictions)


def reset_metrics():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    m_test_loss.reset_states()
    m_test_accuracy.reset_states()




EPOCHS = 50
alpha = 1
nruns = 10  #set to 0 for skip training

#filters=['f150w','f200w','f356w','f444w']
filters = ['F150W', 'F277W', 'F444W']
train = [0,0,1]

data_path = "/n03data/huertas/CANDELS/"
data_COSMOS = "/n03data/huertas/COSMOS-Web/"
loss_object = tf.keras.losses.CategoricalCrossentropy()
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
f_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

m_test_loss = tf.keras.metrics.Mean(name='m_test_loss')
m_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='m_test_accuracy')

conf_train_loss = tf.keras.metrics.Mean(name='c_train_loss')
conf_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='c_train_accuracy')
feature_generator = get_network()
label_predictor = LabelPredictor()
domain_predictor = DomainPredictor()
label_predictor.save_weights(data_COSMOS+"initial_pred.weights")
feature_generator.save_weights(data_COSMOS+"initial_feature.weights")  
domain_predictor.save_weights(data_COSMOS+"initial_domain.weights") 

if os.path.exists(data_path+'image_arrays/CANDELS.npz'):
    print("Loading CANDELS data from saved array")
    data = np.load(data_path+'image_arrays/CANDELS.npz',allow_pickle=True)
    # Access the saved variables
    X = data['stamps']
    label = data['label']    

else:
    X,label = read_CANDELS_data(data_path)
    if WRITE_CANDELS:
        print("Saving CANDELS data")
        np.savez(data_path+'image_arrays/CANDELS.npz', stamps = X, label = label)
for f,tr in zip(filters,train):

    if tr ==0:
        print('skipping training for filter ' + f)
        continue
    
    if os.path.exists(data_COSMOS+'image_arrays/image_arrays_'+f+'.npz'):
        print("Loading saved array with data from filter "+f)
        data = np.load(data_COSMOS+'image_arrays/image_arrays_'+f+'.npz',allow_pickle=True)
        # Access the saved variables
        X_JWST = data['stamps']
        fullvec = data['fullvec']
        idvec = data['idvec']
        fieldvec = data['fieldvec']
        ravec = data['ravec']
        decvec = data['decvec']
        
    else:

        X_JWST,fullvec,idvec,fieldvec,ravec,decvec = read_COSMOS_data([f],data_COSMOS)

        if WRITE:
            print("writing image files for filter "+ str(f))
            np.savez(data_COSMOS+'image_arrays/image_arrays_'+f+'.npz', stamps = X_JWST, fullvec = fullvec, idvec=idvec,fieldvec=fieldvec,ravec=ravec,decvec=decvec)

             

    

    for num in range(nruns):

        tf.keras.backend.clear_session()
        gc.collect()
        label_predictor.load_weights(data_COSMOS+"initial_pred.weights")
        domain_predictor.load_weights(data_COSMOS+"initial_domain.weights")
        feature_generator.load_weights(data_COSMOS+"initial_feature.weights")
        CANDELS_X,label_candels,CANDELS_X_t,label_candels_t,JWST_X,label_JWST,JWST_X_all = create_datasets(X,label,X_JWST)

        all_train_domain_images = np.vstack((CANDELS_X, JWST_X))
        channel_mean = all_train_domain_images.mean((0,1,2))

        train_ds = tf.data.Dataset.from_tensor_slices((CANDELS_X, label_candels)).shuffle(10000).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((CANDELS_X_t, label_candels_t)).batch(32)



        mnist_m_train_ds = tf.data.Dataset.from_tensor_slices((JWST_X,tf.cast(label_JWST, tf.int8))).batch(32)
        mnist_m_test_ds = tf.data.Dataset.from_tensor_slices((JWST_X,tf.cast(label_JWST, tf.int8))).batch(32)


        

        x_train_domain_labels = np.ones([len(label_candels)])
        mnist_m_train_domain_labels = np.zeros([len(label_JWST)])
        all_train_domain_labels = np.hstack((x_train_domain_labels, mnist_m_train_domain_labels))
        all_train_domain_labels = tf.one_hot(all_train_domain_labels, 2).numpy()
        tf.print(all_train_domain_labels)
        domain_train_ds = tf.data.Dataset.from_tensor_slices((all_train_domain_images, tf.cast(all_train_domain_labels, tf.int8))).shuffle(60000).batch(32)
        

        



        for epoch in range(EPOCHS):
            reset_metrics()
        
            for domain_data, label_data in zip(domain_train_ds, train_ds):
            
                try:
                    train_step(label_data[0], label_data[1], domain_data[0], domain_data[1], alpha=alpha)
            
            #End of the smaller dataset
                except ValueError: 
                    pass
            
            for test_data, m_test_data in zip(test_ds,mnist_m_test_ds):
                test_step(test_data[0], test_data[1], m_test_data[0], m_test_data[1])
        
            template = 'Epoch {}, Train Accuracy: {}, Domain Accuracy: {}, Source Test Accuracy: {}, Target Test Accuracy: {}'
            print (template.format(epoch+1,
                                train_accuracy.result()*100,
                                conf_train_accuracy.result()*100,
                                test_accuracy.result()*100,
                                m_test_accuracy.result()*100,))


        label_predictor.save_weights(data_COSMOS+"models/adversarial_asinh_resnet_"+f+"vDR05_1122_shuffle_"+str(num)+".weights")
        feature_generator.save_weights(data_COSMOS+"models/adversarial_asinh_resnet_"+f+"vDR05_1122_shuffle_"+str(num)+".weights")           
        chunk=1000

        sph=[]
        dk=[]
        irr=[]
        bd=[]
        
        n=0    
        while(n<len(JWST_X_all)):
            if n+chunk>len(JWST_X_all):
                p = label_predictor(feature_generator(JWST_X_all[n:]))
            else:    
                p = label_predictor(feature_generator(JWST_X_all[n:n+chunk]))
            n=n+chunk
            print(len(p))
            sph.append(p[:,0])
            dk.append(p[:,1])
            irr.append(p[:,2])
            bd.append(p[:,3])

        if num==0:
            df = pd.DataFrame(list(zip(fullvec,idvec,fieldvec,ravec,decvec,np.concatenate(sph).ravel(),np.concatenate(dk).ravel(),np.concatenate(irr).ravel(),np.concatenate(bd).ravel())),columns =['fullname','id','FIELD', 'ra','dec','sph_0_'+f,'disk_0_'+f,'irr_0_'+f,'bd_0_'+f])  
        else:
            df['sph_'+str(num)+'_'+f]=np.concatenate(sph).ravel()   
            df['disk_'+str(num)+'_'+f]=np.concatenate(dk).ravel()    
            df['irr_'+str(num)+'_'+f]=np.concatenate(irr).ravel()  
            df['bd_'+str(num)+'_'+f]=np.concatenate(bd).ravel() 
        tf.keras.backend.clear_session() 
        gc.collect()
    today = date.today()

    if TRAIN:
        d4 = today.strftime("%b-%d-%Y")        
        df.to_csv(data_COSMOS+"cats/COSMOS-Web_2.0_adversarial_asinh_"+f+"_"+d4+"_4class_shuffle_"+str(nruns)+"_"+str(EPOCHS)+".csv")

