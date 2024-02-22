import argparse
from glob import glob
from datetime import datetime
import numpy as np
import os
import scipy
import astropy.io.fits as pyfits
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.wcs import WCS
import astropy.wcs as wcs
import matplotlib
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn, vstack, hstack
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.nddata import Cutout2D

from astropy.visualization.stretch import SqrtStretch, LogStretch
from astropy.visualization import ImageNormalize, MinMaxInterval, ZScaleInterval
from astropy.visualization import simple_norm 
from astropy import stats

import matplotlib.backends.backend_pdf
    
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#F05A28", "#034693", "#016FB9", "#FF9505","#353531"])
# %matplotlib notebook
from matplotlib.colors import LogNorm
import matplotlib_setup
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import warnings
from astropy.utils.exceptions import AstropyWarning

np.seterr(all='ignore')
warnings.simplefilter('ignore', category=AstropyWarning)

from tools import *

now = datetime.now()
current_time = now.strftime("%H-%M-%S")


parser = argparse.ArgumentParser()
parser.add_argument('--source_ids', nargs='+', type=int, help='List of source ids to make stamps of. IDs need to correspond to the catalog version', required=True)
parser.add_argument('--output_path', type=str, help='Path where to save the output figure', required=True)
parser.add_argument('--catalog_ver', type=str, help='Catalog version, default is v1.6.0', default='v1.6.0')
# parser.add_argument('--obs_window', type=str, help='Which observational window? JAN23 and APR23 are the two options', required=True)
# parser.add_argument('--APR23_tile', type=str, default='A1', help='Plotting a source in APR23 requires to specify the tile ID in which it lies: A{1-10}. Default is A1.')
parser.add_argument('--cutout_size', type=float, default=5, help='Size of the cutouts in arcsec. Default is 5 arcsec')



# makeCutouts.py --source_ids 28833 233175 225676 281448 --output_path '/n17data/shuntov/COSMOS-Web/Catalogs/TestNew' --cutout_size 5


args = parser.parse_args()

source_ids = args.source_ids
output_path = args.output_path
ver = args.catalog_ver
# obs_window = args.obs_window
# tile = args.APR23_tile
cutout_size = args.cutout_size

print('\n Plotting for the following source IDs', source_ids)
print()


############################## CONFIG ###############################

name_SEpp_cat = '/n17data/shuntov/COSMOS-Web/Catalogs/COSMOSWeb_master_v1.6.0-sersic+BD-em_cgs_LePhare_nodupl.fits'

        
# bands_to_plot = ['CFHT-u', 'HSC-g', 'HSC-r', 'HSC-i', 'HST-F814W', 'HSC-z', 'HSC-y', 'UVISTA-Y', 'F115W', 'UVISTA-J', 'F150W', 'UVISTA-H', 'UVISTA-Ks', 'F277W', 'F444W', 'F770W']
bands_to_plot = ['CFHT-u', 'HSC-g', 'HST-F814W', 'HSC-z', 'UVISTA-Y', 'F115W', 'UVISTA-J', 'F150W', 'UVISTA-Ks', 'F277W', 'F444W', 'F770W']
test_idxs = source_ids 
# nameout = f'/n17data/shuntov/COSMOS-Web/Figures/stamps_{ver}.pdf' 
nameout = f'{output_path}_stamps_{ver}.pdf' 


print('\n Making figure for source in catalog', name_SEpp_cat)
print('\n Saving in ', nameout)
print('\n')
######################################################################

fcgs = u.erg/u.s/u.cm/u.cm/u.Hz

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time, '\n')

with fits.open(name_SEpp_cat) as hdu:
    cat0 = hdu[1].data
ID = identity = cat0['id']

tiles=[]
for ll in range(len(source_ids)):
    tt = cat0[np.where(ID==source_ids[ll])]['TILE'][0]
    tiles.append(tt)

print('\n Plotting sources with IDs', source_ids)
print('\n Which are in the tiles', tiles)

sc_all = SkyCoord(cat0['RA_MODEL'], cat0['DEC_MODEL'], unit='deg')



def load_cigale_results(sid):
    with fits.open('/automnt/n17data/arango/CIGALE/runs/run_web/COSMOSWEB_1.6_INPUT/out/results.fits') as hdu:
        cig = hdu[1].data
    return cig[np.where(cig['id'] == np.float(sid))]
def load_cigale_model(sid):
    with fits.open(f'/automnt/n17data/arango/CIGALE/runs/run_web/COSMOSWEB_1.6_INPUT/out/{sid}.0_best_model.fits') as hdu:
        cigale_model = hdu[1].data
    return cigale_model
def load_cigale_results_noagn(sid):
    with fits.open('/automnt/n17data/arango/CIGALE/runs/run_web/COSMOSWEB_1.6_INPUT/no_agn/out/results.fits') as hdu:
        cig = hdu[1].data
    return cig        
def load_cigale_model_noagn(sid):
    with fits.open(f'/automnt/n17data/arango/CIGALE/runs/run_web/COSMOSWEB_1.6_INPUT/no_agn/out/{sid}.0_best_model.fits') as hdu:
        cigale_model = hdu[1].data
    return cigale_model


    

def get_filename(path, filt, key):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if filt in file:
                if key in file:
                    return str(file)

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

    
rms_dict = {'F115W': 9.860080960988429e-32, 'F150W': 7.835102817851673e-32, 'F277W': 4.264199694943563e-32, 'F444W': 4.781897964830027e-32, 'F770W': 6.422508477735181e-31, 'HST-F814W': 6.911436784723074e-32, 'CFHT-u': 1.0855961921345284e-31, 'HSC-g': 6.481630257126643e-32, 'HSC-r': 9.396704648140483e-32, 'HSC-i': 1.1030302326198189e-31, 'HSC-z': 1.6051791754357754e-31, 'HSC-y': 2.900192075471893e-31, 'HSC-NB0816': 2.767802492157106e-31, 'HSC-NB0921': 2.968485155326272e-31, 'HSC-NB1010': 1.5492021104907555e-30, 'UVISTA-Y': 3.743949368070798e-31, 'UVISTA-J': 4.122299369808702e-31, 'UVISTA-H': 5.288675800653872e-31, 'UVISTA-Ks': 6.3978686442199525e-31, 'UVISTA-NB118': 8.55624231852852e-31, 'SC-IA484': 2.438245631287472e-31, 'SC-IA527': 2.5631928002919793e-31, 'SC-IA624': 2.867865033970265e-31, 'SC-IA679': 5.349833149478257e-31, 'SC-IA738': 3.697765368112493e-31, 'SC-IA767': 5.5784177653585204e-31, 'SC-IB427': 3.821443403672672e-31, 'SC-IB505': 3.3837725296289714e-31, 'SC-IB574': 5.603183965699781e-31, 'SC-IB709': 3.9299221388816087e-31, 'SC-IB827': 5.778135545602947e-31, 'SC-NB711': 5.979310643938058e-31, 'SC-NB816': 5.308787029011782e-31}
    
depth_dict = {'F115W': 4.876580622712804e-31, 'F150W': 3.859142545562761e-31, 'F277W': 2.0864886193094827e-31, 'F444W': 2.372572599588078e-31, 'F770W': 3.158883568703061e-30, 'HST-F814W': 3.574309760743438e-31, 'CFHT-u': 4.210738175287924e-31, 'HSC-g': 3.4066114547194664e-31, 'HSC-r': 4.858572939972675e-31, 'HSC-i': 5.786380512168374e-31, 'HSC-z': 8.280942782817936e-31, 'HSC-y': 1.4949296594516324e-30, 'HSC-NB0816': 1.4001444862787465e-30, 'HSC-NB0921': 1.4975402111380154e-30, 'HSC-NB1010': 7.697516918235287e-30, 'UVISTA-Y': 1.7259195925898033e-30, 'UVISTA-J': 1.7692175289238835e-30, 'UVISTA-H': 2.36529770165367e-30, 'UVISTA-Ks': 2.7454526261509744e-30, 'UVISTA-NB118': 3.3886945899817224e-30, 'SC-IA484': 1.2183953941969167e-30, 'SC-IA527': 1.2977212458558441e-30, 'SC-IA624': 1.3673191309116248e-30, 'SC-IA679': 2.6570200193602285e-30, 'SC-IA738': 1.726946589440546e-30, 'SC-IA767': 2.6425943879659974e-30, 'SC-IB427': 1.9291340969782975e-30, 'SC-IB505': 1.6254776021288553e-30, 'SC-IB574': 2.7273592900590886e-30, 'SC-IB709': 1.8402136893141772e-30, 'SC-IB827': 2.6868541933749543e-30, 'SC-NB711': 2.900333936354819e-30, 'SC-NB816': 2.5588452285270337e-30}


def load_imgs(tile):

    if tile == 'JAN':
        name_img_det        = f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-chi2-SWLW_COSMOS-Web_30mas_resc100.fits'
        sci_imas={
            'F115W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F115W_60mas_sci.fits',
            'F150W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F150W_60mas_sci.fits',
            'F277W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F277W_60mas_sci.fits',
            'F444W':      f'/n17data/shuntov/COSMOS-Web/Images_NIRCam_jwst-pipe/v0.004/mosaic-NIRCAM_v0.004_F444W_60mas_sci.fits',
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

    if tile == 'JAN':
        ver = 'v1.9.7'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/{ver}-FF/'

    if 'A' in tile and tile != 'JAN':
        ver = 'v1.5.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/APR-{tile}_{ver}-FF/'

    name_img_part  = path_checkimg+get_filename(path_checkimg, '', '_partition.fits')
    model_imas = {}
    resid_imas = {}

    # print(path_checkimg)
    print(path_checkimg)
    for filt in bands_to_plot:
#         print('\n Filter', filt, '\n')
        if filt in filters_FF:
            if tile == 'JAN':
                f = get_filename(path_checkimg, filters_translate[filt], 'v1.9.71_model')
            else:
                f = get_filename(path_checkimg, filters_translate[filt], 'model')
        else:
            f = get_filename(path_checkimg, filters_translate[filt], 'model')
#         if filt == 'F770W':
#             os.system(f'rm {path_checkimg}*miri*v0_01*')
#             f = get_filename(path_checkimg, filters_translate[filt], 'model')
#         print('Model File',f)
        try: 
            model_imas[filt] = path_checkimg + f
        except TypeError as err:
            print('Error for', tile, ':', err)
            
        if filt in filters_FF:
            if tile == 'JAN':
                f = get_filename(path_checkimg, filters_translate[filt], 'v1.9.71_resid')
            else:
                f = get_filename(path_checkimg, filters_translate[filt], 'resid')
        else:
            f = get_filename(path_checkimg, filters_translate[filt], 'resid')
#         print('Resid File',f)
        try:
            resid_imas[filt] = path_checkimg + f
        except TypeError as err:
            print('Error for', tile, ':', err, 'and filter', filt)
            print('\n There seems to be an error for some galaxies in JAN tile. Remove them for the list \n')
        
            
    


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

def filter_image(img, filter_type='box2d', kernel_size=2):
    '''
    filter type can be:
        - box2d
        - gaussian
    '''
    from astropy.convolution import convolve
    
    if filter_type == 'box2d':
        from astropy.convolution import Box2DKernel
        kernel = Box2DKernel(kernel_size, mode='oversample')
    if filter_type == 'gaussian':
        from astropy.convolution import Gaussian2DKernel
        kernel = Gaussian2DKernel(x_stddev=1, x_size=kernel_size, y_size=kernel_size, mode='oversample')
        
    img_filt = convolve(img, kernel)
    
    return img_filt


def plot_source_diagnostic(sid, w, ra_cent, dec_cent, img_detec, img_part, img_detec_c20, img_list, model_list, resid_list,
                          photometry_list, wavelengths, photometry_labels, 
                          lambda_sed, mag_sed, lambda_mag, magmodel_sed, magmodel_mes, magmodelerr_mes, models_info, zpdf,
                          compar_photometry_list, compar_wavelengths, isFilter=True, show=True):
    
    from astropy.visualization.stretch import SqrtStretch, LogStretch
    from astropy.visualization import ImageNormalize, MinMaxInterval, ZScaleInterval
    from astropy.visualization import simple_norm 
    from astropy import stats
    

    
    cmap1 =  'bone_r'  #'inferno'
    cmap2 =  'PuOr' #'PuOr'    #'inferno'

    img_part = img_part.astype('float')
    img_part[img_part==0] = np.nan
    
    ncol = len(img_list)+1 
    nrow = 3 + 2
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4))#, subplot_kw={'projection': w}), dpi=200
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
#     fig.suptitle('JWST PSFs')
    fig.suptitle(f'ID = {sid}', fontsize=43)


    sc_1 = SkyCoord(ra_cent, dec_cent, unit='deg')
    sc_pix_1 = skycoord_to_pixel(sc_1, w[0])
    
    sc_pix_all = skycoord_to_pixel(sc_all, w[0])
    
    ccond = (sc_all.dec.value < dec_cent+arcsec_cut/2/3600) & (sc_all.dec.value > dec_cent-arcsec_cut/2/3600) & (sc_all.ra.value < ra_cent+arcsec_cut/2/3600) & (sc_all.ra.value > ra_cent-arcsec_cut/2/3600) 
    sc_pix_close_ra = sc_pix_all[0][ccond] 
    sc_pix_close_dec = sc_pix_all[1][ccond] 
    
    
    axs[0,0].remove()
    axs[0,0] = fig.add_subplot(nrow, ncol, 1)#, projection=w)
#     lon = axs[0,0].coords[0]
#     lat = axs[0,0].coords[1]
    
#     lon.set_axislabel('RA')
#     lat.set_axislabel('Dec')
#     lat.set_ticklabel(rotation=90)
# #     lat.set_ticks([dec_cent, dec_cent+0.001] * u.deg, size=16)
#     lat.set_ticklabel(rotation=90)
# #     axs[0,0].tick_params(axis='both', labelsize=16)
# #     lon.set_ticks([ra_cent, ra_cent+0.001] * u.deg, size=16)
# #     axs[0,0].tick_params(axis='both', labelsize=16)
# #     lon.set_major_formatter('hh:mm:ss') ### THIS
# #     lat.set_major_formatter('dd:mm:ss') ### THIS
#     lon.set_major_formatter('d.ddd') ### THIS
#     lat.set_major_formatter('d.ddd') ### THIS

    
    if isFilter:
        img_detec = filter_image(img_detec, kernel_size=3)
    axs[0,0].set_title('Detection', fontsize=30)
    try:
        mmin_detec,mmax_detec = MinMaxInterval().get_limits(img_detec) # ZScaleInterval(contrast=0.25)
    except ValueError as err:
        mmin_detec = 0; mmax_detec = 2
    norm = ImageNormalize(stretch=SqrtStretch(), vmin=mmin_detec*0.4, vmax=mmax_detec*2)
    axs[0,0].imshow(img_detec, norm=norm, cmap=cmap1, origin='lower')
    axs[2,0].set_title('Detection COSMOS2020', fontsize=30)
    
    try:
        mmin_c20,mmax_c20 = MinMaxInterval().get_limits(img_detec_c20) # ZScaleInterval(contrast=0.25)
    except ValueError as err:
        mmin_c20 = 0; mmax_c20 = 2
    norm = ImageNormalize(stretch=SqrtStretch(), vmin=mmin_c20*0.5, vmax=mmax_c20*2)
    axs[2,0].imshow(img_detec_c20, norm=norm, cmap=cmap1, origin='lower')
    axs[1,0].set_title('Partition', fontsize=30)
    axs[1,0].imshow(img_part, cmap='prism_r', vmin=0, origin='lower')
    
    axs[1,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=5, color='w', markeredgewidth=3) 
    axs[1,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=5, color='k', markeredgewidth=1) 
#     axs[1,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=28, color='k', markeredgewidth=2, markerfacecolor='none', alpha=0.99) 
#     axs[1,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=28, color='w', markeredgewidth=1, markerfacecolor='none', alpha=0.99) 
    axs[1,0].axvline(len(img_part)//2, ls='dashed', c='k', lw=1)
    axs[1,0].axhline(len(img_part)//2, ls='dashed', c='k', lw=1)
    
#     axs[0,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=28, color='k', markeredgewidth=2, markerfacecolor='none', alpha=0.7) 
#     axs[0,0].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=28, color='w', markeredgewidth=1, markerfacecolor='none', alpha=0.7) 
    
    for i,j in zip(range(1,ncol), range(0, ncol)):
        sc_pix_1 = skycoord_to_pixel(sc_1, w[j+1])
        sc_pix_all = skycoord_to_pixel(sc_all, w[j+1])
        ccond = (sc_all.dec.value < dec_cent+arcsec_cut/2/3600) & (sc_all.dec.value > dec_cent-arcsec_cut/2/3600) & (sc_all.ra.value < ra_cent+arcsec_cut/2/3600) & (sc_all.ra.value > ra_cent-arcsec_cut/2/3600) 
        sc_pix_close_ra = sc_pix_all[0][ccond] 
        sc_pix_close_dec = sc_pix_all[1][ccond] 
            
        if isFilter:
            img_list[j] = filter_image(img_list[j], kernel_size=3)
            model_list[j] = filter_image(model_list[j], kernel_size=3)
        axs[0,i].set_title(photometry_labels[j], fontsize=30)

        try:
            mmin_img,mmax_img = MinMaxInterval().get_limits(img_list[-3]) # ZScaleInterval(contrast=0.25)
#             mmin_img,mmax_img = MinMaxInterval().get_limits(img_detec) # ZScaleInterval(contrast=0.25)
        except ValueError as err:
            mmin_img = 0; mmax_img = 2
        norm = ImageNormalize(stretch=SqrtStretch(), vmin=mmin_img*0.4, vmax=mmax_img*2)
        try:
            axs[0,i].imshow(img_list[j], norm=norm, cmap=cmap1, origin='lower')
            ######### mark sources
            # axs[0,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='w', markeredgewidth=3) 
            # axs[0,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='k', markeredgewidth=1) 
            # axs[0,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='w', markeredgewidth=3) 
            # axs[0,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='k', markeredgewidth=1) 
        except IndexError as err:
            axs[0,i].imshow(np.ones_like(img_list[j-1]), norm=simple_norm(np.ones_like(img_list[j-1]), 'sqrt', percent=99.7, min_cut=0, max_cut=mcut), cmap=cmap1, origin='lower')
            
#         norm = ImageNormalize(stretch=SqrtStretch(), vmin=mmin_img, vmax=mmax_img)
        try:
#             modimg = model_list[j]-mmin_img
            axs[1,i].imshow(model_list[j], norm=norm, cmap=cmap1, origin='lower')
            ######### mark sources
#             axs[1,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='w', markeredgewidth=3) 
#             axs[1,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='k', markeredgewidth=1) 
#             axs[1,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='w', markeredgewidth=3) 
#             axs[1,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='k', markeredgewidth=1) 
        except IndexError as err:
            axs[0,i].imshow(np.ones_like(img_list[j-1]), norm=simple_norm(np.ones_like(img_list[j-1]), 'sqrt', percent=99.7, min_cut=0, max_cut=mcut), cmap=cmap1, origin='lower')

        try:
#             _, _, stdev = stats.sigma_clipped_stats(resid_list[j], sigma=3) #, maxiters=3)
            stdev = np.nanstd(resid_list[j])
            axs[2,i].imshow(resid_list[j], norm='linear', 
                         cmap=cmap2, origin='lower', vmin=-6*stdev, vmax=6*stdev)
            ######### mark sources
#             axs[2,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='w', markeredgewidth=3) 
#             axs[2,i].plot(sc_pix_all[0][ccond], sc_pix_all[1][ccond], 'o', ms=5, color='k', markeredgewidth=1) 
#             axs[2,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='w', markeredgewidth=3) 
#             axs[2,i].plot(sc_pix_1[0], sc_pix_1[1], '+', ms=11, color='k', markeredgewidth=1) 
        except IndexError as err:
            axs[0,i].imshow(np.ones_like(img_list[j-1]), norm=simple_norm(np.ones_like(img_list[j-1]), 'sqrt', percent=99.5, min_cut=0), cmap=cmap1, origin='lower')
            
#         axs[0,i].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=5, color='w', markeredgewidth=3) 
#         axs[0,i].plot(sc_pix_close_ra, sc_pix_close_dec, 'o', ms=5, color='k', markeredgewidth=1) 

#         axs[0,i].invert_yaxis()
#         axs[1,i].invert_yaxis()
#         axs[2,i].invert_yaxis()
#         axs[1,0].invert_yaxis()
        
#     lat.set_axislabel(' ')
#     lon.set_axislabel(' ')
    for o in [0,1,2]:
        for oo in range(ncol):
            axs[o, oo].set_xticks([]) #set_axis_ticks_off()
            axs[o, oo].set_yticks([])

#     axs[0,ncol-1].set_title(f'ID = {sid}', fontsize=30)
    
#     y = photometry_list[0].reshape(len(photometry_list[0]),)
#     yerr =  photometry_list[1].reshape(len(photometry_list[1]),)
#     masky = np.where(yerr<0, 99, y)
#     maskyerr = np.where(yerr<0, 99, yerr)
#     axs[0,ncol-1].errorbar(wavelengths, masky, yerr=maskyerr,  
#         fmt='o', capthick=2, label='MODEL')
    
#     y = compar_photometry_list[0].reshape(len(compar_photometry_list[0]),)
#     yerr =  compar_photometry_list[1].reshape(len(compar_photometry_list[1]),)
#     masky = np.where(yerr<0, 99, y)
#     maskyerr = np.where(yerr<0, 99, yerr)
#     axs[0,ncol-1].errorbar(wavelengths, masky, yerr=maskyerr,   
#         fmt='none', capthick=2, color='navy')
    
#     axs[0,ncol-1].plot(compar_wavelengths, compar_photometry_list[0].reshape(len(compar_photometry_list[0]),), 'o', mfc='none', label='APER', ms=7, color='navy')
#     axs[0,ncol-1].set_ylim(34.0, 16.5)
#     axs[0,ncol-1].set_xlim(0.3, 10.1)
#     axs[0,ncol-1].set_xscale('log')
#     axs[0,ncol-1].set_axis_on()
#     axs[0,ncol-1].set_xlabel('Wavelength [$\\mu m$]', fontsize=26)
#     axs[0,ncol-1].set_ylabel('Magnitude [AB]', fontsize=26)
#     axs[0,ncol-1].legend(fontsize=14)
#     axs[0,ncol-1].minorticks_on()
    
    ######################################### PLOT LARGE SED ###################################
    gs = axs[3, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[3, 0:]:
        ax.remove()
    for ax in axs[4, 0:]:
        ax.remove()
    axbig = fig.add_subplot(gs[3:, 0:-4])
    axbig2 = fig.add_subplot(gs[3:4, -4:])
    
    
#     y = photometry_list[0].reshape(len(photometry_list[0]),)
#     yerr =  photometry_list[1].reshape(len(photometry_list[1]),)
    ################## plots magnitudes
    if True:
        # only the reliable obs mag will be plotted:
        em = magmodelerr_mes
        lf = lambda_mag
        mag = magmodel_mes
        
        em=em*2.
        mag1=mag[(mag>0.) & (mag<35) & (em>-3) & (dlf>50)]
        em1=em[(mag>0.) & (mag<35) & (em>-3) & (dlf>50)]
        lf1=lf[(mag>0.) & (mag<35) & (em>-3) & (dlf>50)]

        if(len(mag1>0)):
            ymin=max(mag1+2.)
            ymax=min(mag1-4.)
        else:
            ymin=10 
            ymax=20
        if ymin>60:  
            ymin=30

        ic=(em1>=0.) & (em1<2.)
        lf2=lf1[ic]
        mag2=-.4*(mag1[ic]-23.91) 
        em2=0.4*em1[ic]
        # low S/N bands:  
        ic2=(em1>=2.) & (em1<8.)
        lf2b=lf1[ic2]
        mag2b=-.4*(mag1[ic2]-23.91) 
        em2b=0.4*em1[ic2]
        
        axbig.errorbar(lf2, mag2, yerr=em2, fmt='o', capthick=5, label='Measured', linewidth=4, ms=21, color='navy')
        axbig.errorbar(lf2b, mag2b, yerr=em2b, fmt='o', capthick=5, linewidth=4, ms=21, color='navy', alpha=0.2)

        
    ######### plots fluxes
    if False:
        sph = np.zeros(len(band_lambda_micron))
        spher = np.zeros(len(band_lambda_micron))
        lambdaa = np.zeros(len(band_lambda_micron))
        for bi,bb in enumerate(band_lambda_micron):
            sph[bi] = (cat0[np.where(cat0['id']==sid[0])][f'FLUX_MODEL_{bb}'] * fcgs).to(u.uJy).value
            spher[bi] = (cat0[np.where(cat0['id']==sid[0])][f'FLUX_ERR_MODEL_{bb}'] * fcgs).to(u.uJy).value*1.05
            lambdaa[bi] = band_lambda_micron[bb]
            
            #### set flux to -99 if lower than the depth
#             if sph[bi] < (rms_dict[bb]*fcgs).to(u.uJy).value:
#                 sph[bi] = -99
    
        masky = np.where(spher>sph, 0, sph)
        maskyerr = np.where(spher>sph, 99, spher)
        masky = np.where(maskyerr<0, 0, masky)
        maskyerr = np.where(maskyerr<0, 99, maskyerr)
        masky2 = sph[np.where(spher>sph)]
        lambda_mask = lambdaa[np.where(spher>sph)]
#         masky = np.where(sph < [(depth_dict[bb]*fcgs).to(u.uJy).value for bb in band_lambda_micron], -99, sph)
#         maskyerr = np.where(sph < [(depth_dict[bb]*fcgs).to(u.uJy).value for bb in band_lambda_micron], 99, spher)
        masky3 = sph[np.where(sph < [(depth_dict[bb]*fcgs).to(u.uJy).value for bb in band_lambda_micron])]
        lambda_mask3 = lambdaa[np.where(sph < [(depth_dict[bb]*fcgs).to(u.uJy).value for bb in band_lambda_micron])]
        
        yerr = maskyerr #/masky
        # yerr = [np.log10(masky + maskyerr)-np.log10(masky), np.log10(masky)-np.log10(masky - maskyerr)]
        axbig.errorbar(lambdaa, (masky), yerr=yerr, fmt='o', capthick=5, label='Measured', linewidth=4, ms=25, color='navy')
#         axbig.errorbar(lambda_mask, masky2, yerr=0, fmt='o', capthick=5, linewidth=4, ms=25, color='navy', alpha=0.2)
        axbig.errorbar(lambda_mask3, (masky3), yerr=np.abs((masky3*0.75)), uplims=np.ones_like(masky3), fmt='.', capsize=10, capthick=7, linewidth=7, ms=0, color='navy', alpha=0.3)

            
    # lsort = np.argsort(lambda_sed)
    axbig.plot(lambda_sed[0], (mag_sed[0]), lw=8, alpha=0.9, label='LePhare best fit', color='#f6511d')
    # axbig.plot(lambda_sed[1], (mag_sed[1]), lw=5, alpha=0.5, label='LePhare GAL-2', color=plt.cm.magma(0.5))
    axbig.plot(lambda_sed[2], (mag_sed[2]), lw=4, alpha=0.65, label='LePhare QSO', color=plt.cm.magma(0.55))
    axbig.plot(lambda_sed[3], (mag_sed[3]), lw=2, alpha=0.12, label='LePhare Star', color=plt.cm.magma(0.0))
    axbig.plot(lambda_mag, (magmodel_sed), 'o', mfc='none', ms=30, color='#f6511d', markeredgewidth=5, alpha=0.9)

    try:
        cigale_model = load_cigale_model(sid[0])
        axbig.plot(cigale_model['wavelength']/1000, np.log10((cigale_model['Fnu']*1.0e3)), #.to(u.ABmag), 
            lw=6, alpha=0.7, color='lightseagreen', label='CIGALE')
    except:
        pass


    
    # (cat0[np.where(cat0['id']==sid[0])]['RADIUS']*3600 < 0.03) &
    # & (cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F115W']-cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F150W'] < 0.8)
#     (cat0[np.where(cat0['id']==sid[0])]['FLUX_APER_F277W'][:,2]/cat0[np.where(cat0['id']==sid[0])]['FLUX_APER_F277W'][:,1] < 2.0) &
    if False:
        if ((cat0[np.where(cat0['id']==sid[0])]['AXRATIO']>0.7) | (cat0[np.where(cat0['id']==sid[0])]['RADIUS']*3600 < 0.1)) & (cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F277W']-cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F444W'] > 1.0) :
            axbig.text(11.22,0.002,'LRD', color='red', fontsize='52')#, bbox=dict(facecolor='white', alpha=0.3))
        axbig.text(11.22,0.0008, 'F277W-F444W='+str( np.around(cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F277W']-cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F444W'],2) ), color='k', fontsize='29')#, bbox=dict(facecolor='white', alpha=0.3))
        axbig.text(11.22,0.0002, 'F115W-F150W='+str( np.around(cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F115W']-cat0[np.where(cat0['id']==sid[0])]['MAG_MODEL_F150W'],2) ), color='k', fontsize='29')#, bbox=dict(facecolor='white', alpha=0.3))
        axbig.text(11.22,0.00008, '$F(<0.5)/F(<0.25)$='+str( np.around(cat0[np.where(cat0['id']==sid[0])]['FLUX_APER_F277W'][:,2]/cat0[np.where(cat0['id']==sid[0])]['FLUX_APER_F277W'][:,1],2) ), color='k', fontsize='29')#, bbox=dict(facecolor='white', alpha=0.3))
        axbig.text(11.22,0.00002, 'AXRATIO='+str( np.around(cat0[np.where(cat0['id']==sid[0])]['AXRATIO'],2) ), color='k', fontsize='29')#, bbox=dict(facecolor='white', alpha=0.3))
        axbig.text(11.22,0.02,'$R_{eff}=$'+str(np.around(cat0['RADIUS'][np.where(cat0['Id']==identity)][0]*3600, 3))+' arcsec', color='k', fontsize='46')#, bbox=dict(facecolor='white', alpha=0.3))
    #     0-Type 1-Nline 2-Model 3-Library 4-Nband  5-Zphot 6-Zinf 7-Zsup 8-Chi2  9-PDF  10-Extlaw 11-EB-V 12-Lir 13-Age  14-Mass 15-SFR 16-SSFR
    #     axbig.text(11.22,18,'$z_{phot}$ = '+models_info[5][:-3], color='k', fontsize='46')#, bbox=dict(facecolor='white', alpha=0.3))
    
    
    axbig.text(11.22,-0.4*(ymin-23.91)+1.6,'$z_{phot}$ = '+str(cat0['LP_zPDF'][np.where(cat0['Id']==identity)][0]), color='k', fontsize='42')#, bbox=dict(facecolor='white', alpha=0.3))
    axbig.text(11.22,-0.4*(ymin-23.91)+1.1,'log($M_{\star}/M_{\odot}$) = '+str(np.around(cat0['LP_mass_med_PDF'][np.where(cat0['Id']==identity)][0],2)), color='k', fontsize='42')#, bbox=dict(facecolor='white', alpha=0.3))
#     axbig.text(11.22,22,'log($M_{\star}/M_{\odot}$) = '+models_info[14][:-2], color='k', fontsize='46')#, bbox=dict(facecolor='white', alpha=0.3))
    axbig.text(11.22,-0.4*(ymin-23.91)+0.6,'log(sSFR/yr$^{-1}$) = '+models_info[16][:-3], color='k', fontsize='42')#, bbox=dict(facecolor='white', alpha=0.3))
    axbig.text(11.22,-0.4*(ymin-23.91)+0.1,'E(B-V) = '+models_info[11], color='k', fontsize='42')#, bbox=dict(facecolor='white', alpha=0.3))
    
    axbig.set_ylim(-0.4*(ymin-23.91),-0.4*(ymax-23.91)) #(np.log10(0.0001), np.log10(200))
    axbig.set_xlim(0.33, 10.9)
#     plt.setp(axbig, xticks=[0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9], 
#         yticks=[1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e-2],
#         yticklabels=[(1.0e-4*u.uJy).to(u.ABmag).value, (1.0e-3*u.uJy).to(u.ABmag).value, (1.0e-2*u.uJy).to(u.ABmag).value, (1.0e-1*u.uJy).to(u.ABmag).value, (1.0e0*u.uJy).to(u.ABmag).value, (1.0e1*u.uJy).to(u.ABmag).value, (1.0e-2*u.uJy).to(u.ABmag).value])
    axbig.set_xscale('log') 
    # axbig.set_yscale('log')
    axbig.set_axis_on() 
    axbig.set_xlabel('Wavelength [$\\mu m$]', fontsize=43)
    axbig.set_ylabel('Magnitude [AB]', fontsize=43)
    axbig.legend(fontsize=30, loc='lower right', ncol=3)#, bbox_to_anchor=(0, 0), ncol=1, frameon=False)
    axbig.minorticks_on()
    axbig.tick_params(axis='both', which='major', labelsize=39, length=30, width=3)
    axbig.tick_params(axis='both', which='minor', labelsize=39, length=15, width=3)
    axbig.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    axbig.xaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    # axbig.yaxis.set_ticks([np.log10((18.5*u.ABmag).to(u.uJy).value), np.log10((20*u.ABmag).to(u.uJy).value), np.log10((21.5*u.ABmag).to(u.uJy).value), np.log10((23*u.ABmag).to(u.uJy).value), np.log10((24.5*u.ABmag).to(u.uJy).value), np.log10((26*u.ABmag).to(u.uJy).value), np.log10((27.5*u.ABmag).to(u.uJy).value), np.log10((29*u.ABmag).to(u.uJy).value), np.log10((30.5*u.ABmag).to(u.uJy).value), np.log10((32*u.ABmag).to(u.uJy).value)], labels= [18.5, 20.0, 21.5, 23.0, 24.5, 26.0, 27.5, 29.0, 30.5, 32.0])
    axbig.xaxis.set_ticks([0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9], labels= [0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9])
    axbig.grid(alpha=0.7)
    ######################################################################################
    
    
    ########################## Photo-z PDF inset
#     axins2 = inset_axes(axbig, width="20%", height="10%", loc=0)
#     axins2.plot(zpdf[0,:], zpdf[1,:], lw=2.4, alpha=0.5, color='navy')
#     axins2.set_xlabel('$z_{phot}$')
    axins2 = axbig2 #plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    # ip = InsetPosition(axbig, [0.025,0.7,0.4,0.3])
    # axins2.set_axes_locator(ip)
    axins2.plot(zpdf[0,:], zpdf[1,:], lw=4.4, alpha=0.9, color='navy')
    axins2.axvline(cat0['LP_zPDF'][np.where(cat0['Id']==identity)][0], lw=4, color='tomato')
    axins2.axvspan(cat0['LP_zPDF_l68'][np.where(cat0['Id']==identity)][0], cat0['LP_zPDF_u68'][np.where(cat0['Id']==identity)][0], alpha=0.3, color='tomato')
    axins2.set_xlabel('zPDF',fontsize=30)

    
    
    axs[0,1].set_ylabel('Image', fontsize=30)
    axs[1,1].set_ylabel('Model', fontsize=30)
    axs[2,1].set_ylabel('Residual', fontsize=30)
    
    
    
#     axs[1,0].set_axis_off()
#     axs[2,0].set_axis_off()
#     axs[1,2].set_axis_off()
#     axs[2,2].set_axis_off()
#     axs[1,ncol-1].set_axis_off()
#     axs[2,ncol-1].set_axis_off()

#     plt.tight_layout() 
#     plt.savefig(f'/home/shuntov/SMACS0723/SE++/Figures/plot_source-id-{sid}_diagnostic.png', bbox_inches='tight')
    if show==True:
        plt.show()
    if show==False:
        plt.close()
    
    return fig







import math

####################################################
zpdff = {}
lg=[]; mg=[]
lg2=[]; mg2=[]
lg3=[]; mg3=[]
lg4=[]; mg4=[]
lambda_mag = []; magmodel_mes = []; magmodelerr_mes = []; magmodel_sed = []
models_info_=[]
for k in range(len(source_ids)):
    
    splitId=math.modf(float(source_ids[k])/100000.)

    spec_filename = '/n07data/ilbert/COSMOS-Web/photoz_MASTER_1.6.0/spec/SPEC'+str(round(splitId[1]))+'/' #Id340000.spec
    ### Open .spec file[s] and read the parameters
    fsp=open(spec_filename+f'Id{source_ids[k]}.spec','r')

    bid=fsp.readline()    #header1
    line=fsp.readline()
    line=line.split()
    id=line[0]; zspec=line[1]; zphot=float(line[2])
    #z68low=float(line[3]); z68hig=float(line[4])

    bid=fsp.readline()    #header2
    line=fsp.readline()
    line=line.split()
    nfilt=int(line[1])

    bid=fsp.readline()    #header3
    line=fsp.readline()
    line=line.split()
    npdf=int(line[1])

    bid=fsp.readline()  
    #header4:  Type Nline Model Library Nband    Zphot Zinf Zsup Chi2  PDF     Extlaw EB-V Lir Age  Mass SFR SSFR
    
    models_info=[]
    for i in range(6):
        line=fsp.readline()
        model=line.split()    
        models_info.append(model)
        if i == 0:
            models_info_.append(model)

    

    # Read observed mag, err_mag, filters' lambda_eff and width, models' mag
    mag=np.zeros(nfilt); em=np.zeros(nfilt); 
    lf=np.zeros(nfilt); dlf=np.zeros(nfilt); 
    mmod=np.zeros(nfilt); mfir=np.zeros(nfilt); mphys=np.zeros(nfilt)
    for i in range(nfilt):
        line=fsp.readline(); line=line.split() 
        mag[i]=float(line[0]); em[i]=float(line[1]);
        lf[i]=float(line[2]); dlf[i]=float(line[3]); 
        mmod[i]=float(line[4]); mfir[i]=float(line[5]);
        mphys[i]=float(line[6]); mmod[i]=float(line[8])

    #convert mag(AB syst.) in log(flux)
    ibad=np.where((mmod<=0) | (mmod>45))
    mmod=(-0.4*(mmod-23.91))  # uJy
    mmod[ibad]=-10.
    ibad=np.where((mphys<=0) | (mphys>45))
    mphys= (-0.4*(mphys-23.91))  # uJy
    mphys[ibad]=-10.
    ibad=np.where((mfir<=0) | (mfir>45))
    mfir= (-0.4*(mfir-23.91))  # uJy
    mfir[ibad]=-10.

    magmodel_sed.append(mmod)
    magmodel_mes.append(mag)
    magmodelerr_mes.append(em)
    lambda_mag.append(lf/10000)
    
    
    zpdf=np.zeros([3,npdf])
    for i in range(npdf):
        line=fsp.readline()
        zpdf[:,i]=np.array(line.split())
    
    zpdff[k] = zpdf

    # Read spectra [lambda(A), Mag(AB)]
    # convert in log(F_nu) = -0.4*(mab-23.91) [uJy]
    m=0 # for best-fit sed
    nline=int(models_info[m][1])
    bid=np.zeros([2,nline])
    if nline>0:
        for i in range(nline): 
            line=fsp.readline()
            bid[:,i]=np.array(line.split())
            bid[1,i]=(-0.4*(bid[1,i]-23.91))
            if (bid[1,i]>35):
                bid[1,i]=-10.
            else:
                bid[1,i]= bid[1,i] #-0.4*(bid[1,i]-23.91)
            np.delete(bid[1,i], np.where(bid[1,i]>35))
            np.delete(bid[0,i], np.where(bid[1,i]>35))
    lg.append(bid[0,:]/10000.)
    mg.append(bid[1,:])
    
    m=1 # 
    nline=int(models_info[m][1])
    bid=np.zeros([2,nline])
    if nline>0:
        for i in range(nline): 
            line=fsp.readline()
            bid[:,i]=np.array(line.split())
            bid[1,i]=(-0.4*(bid[1,i]-23.91))
            if (bid[1,i]>35):
                bid[1,i]=-10.
            else:
                bid[1,i]= bid[1,i] #-0.4*(bid[1,i]-23.91)
            np.delete(bid[1,i], np.where(bid[1,i]>35))
            np.delete(bid[0,i], np.where(bid[1,i]>35))
    lg2.append(bid[0,:]/10000.)
    mg2.append(bid[1,:])
    
    m=4 # for best-fit sed
    nline=int(models_info[m][1])
    bid=np.zeros([2,nline])
    if nline>0:
        for i in range(nline): 
            line=fsp.readline()
            bid[:,i]=np.array(line.split())
            bid[1,i]=(-0.4*(bid[1,i]-23.91))
            if (bid[1,i]>35):
                bid[1,i]=-10.
            else:
                bid[1,i]= bid[1,i] #-0.4*(bid[1,i]-23.91)
            np.delete(bid[1,i], np.where(bid[1,i]>35))
            np.delete(bid[0,i], np.where(bid[1,i]>35))
    lg3.append(bid[0,:]/10000.)
    mg3.append(bid[1,:])
    
    m=5 # for best-fit sed
    nline=int(models_info[m][1])
    bid=np.zeros([2,nline])
    if nline>0:
        for i in range(nline): 
            line=fsp.readline()
            bid[:,i]=np.array(line.split())
            bid[1,i]=(-0.4*(bid[1,i]-23.91))
            if (bid[1,i]>35):
                bid[1,i]=-10.
            else:
                bid[1,i]= bid[1,i] #-0.4*(bid[1,i]-23.91)
            np.delete(bid[1,i], np.where(bid[1,i]>35))
            np.delete(bid[0,i], np.where(bid[1,i]>35))
    lg4.append(bid[0,:]/10000.)
    mg4.append(bid[1,:])
            

    fsp.close() 
    

####################################################









########################################## DO THE PLOTTING ###################################

pdf = matplotlib.backends.backend_pdf.PdfPages(f'{nameout}')
for j in range(len(test_idxs)):
    
    
    i = np.where(ID == test_idxs[j])    
    
    ra_cent  = cat0['RA_MODEL'][i]
    dec_cent = cat0['DEC_MODEL'][i]
    try:
        identity = cat0['id'][i]
    except:
        identity = cat0['source_id_1'][i]
    
    print('-----------------', identity, '--------------------')
    print((ra_cent*u.deg), (dec_cent*u.deg))
    
    name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate = load_imgs(tiles[j])
    
    ################## define image normalizations 
#     image_data = fits.getdata(name_img_det, ext=0)    
#     mmin_detec,mmax_detec = ZScaleInterval(contrast=0.15).get_limits(image_data)
# #     mmin_detec = 0

#     image_data = fits.getdata(imgname_chi2_c20, ext=0)    
#     mmin_c20,mmax_c20 = ZScaleInterval(contrast=0.15).get_limits(image_data)
# #     mmin_c20 = 0

#     mmin_img = []; mmax_img = []
#     for fff in bands_to_plot:
#         try:
#             image_data = fits.getdata(sci_imas[fff], ext=0)    
#         except:
#             image_data = fits.getdata(sci_imas[fff], ext=1)    
#         mmin,mmax = ZScaleInterval(contrast=0.15).get_limits(image_data)
#         mmin_img.append(mmin)
#         mmax_img.append(mmax)
#     norm_res = 'linear'
    
#     image_data = fits.getdata(resid_imas['F277W'], ext=0)
#     _, _, stdev = stats.sigma_clipped_stats(image_data[::100,::100], sigma=3, maxiters=3)
    

    wavelengths = [band_lambda_micron[filt] for filt in bands_to_plot]
    
    try:
        photometry_list = np.asarray([ [cat0[f'MAG_MODEL_{filt}'][i] for filt in bands_to_plot],
                            [cat0[f'MAG_ERR_MODEL_{filt}'][i] for filt in bands_to_plot]])
    except:
        photometry_list = np.asarray([ [cat0[f'MAG_MODEL_{filt}'][i] for filt in bands_to_plot],
                            [cat0[f'MAG_MODEL_{filt}_err'][i] for filt in bands_to_plot]])
    try:
        compar_photometry_list = np.asarray([ [cat0[f'MAG_APER_{filt}'][i,3][0] for filt in bands_to_plot],
                        [cat0[f'MAG_ERR_APER_{filt}'][i,3][0] for filt in bands_to_plot]])
#         print('Using APER photometry as comparison')
    except:
        compar_photometry_list = np.asarray([ [cat0[f'MAG_AUTO_{filt}'] for filt in bands_to_plot],
                        [cat0[f'MAG_ERR_AUTO_{filt}'][i] for filt in bands_to_plot]])
#         print('Using AUTO photometry as comparison')
    
#     np.asarray([ [cat0[f'MAG_APER_{filt}'][i,3][0] for filt in bands_to_plot],
#                         [cat0[f'MAG_ERR_APER_{filt}'][i,3][0] for filt in bands_to_plot]])
    


    arcsec_cut = cutout_size

    img_detec, w = image_make_cutout(name_img_det, ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=True)
    img_detec_c20, _ = image_make_cutout(imgname_chi2_c20, ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=True)
    img_parti = image_make_cutout(name_img_part, ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=None)

    img_list =[image_make_cutout(sci_imas[filt], ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=None) 
               for filt in bands_to_plot]
    
    w_list = [w]
    for filt in bands_to_plot:
        _, www = image_make_cutout(sci_imas[filt], ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=True)
        
        w_list.append(www)

    model_list =[image_make_cutout(model_imas[filt], ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=None) 
               for filt in bands_to_plot]
    
    resid_list =[image_make_cutout(resid_imas[filt], ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=None) 
               for filt in bands_to_plot]
                                 
    
    fig = plot_source_diagnostic(identity, w_list, ra_cent, dec_cent, img_detec, img_parti, img_detec_c20, img_list, model_list, resid_list,
                              photometry_list, wavelengths, bands_to_plot, 
                                 [lg[j], lg2[j], lg3[j], lg4[j]], [mg[j], mg2[j], mg3[j], mg4[j]], lambda_mag[j], magmodel_sed[j], magmodel_mes[j], magmodelerr_mes[j], models_info_[j], zpdff[j],
                              compar_photometry_list, wavelengths, isFilter=False, show=False)
    
    
    

#     fig.savefig() 
    pdf.savefig(fig)
pdf.close()










