import os
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import Planck13 as cosmo
from matplotlib.backends.backend_pdf import PdfPages
import aplpy
from datetime import date
import astropy.wcs as wcs


data_COSMOS = '/n03data/huertas/COSMOS-Web/cats'
root_COSMOS= '/n03data/huertas/COSMOS-Web/'
f='f150w'



#COSMOS files
cosmos_f150w_fn='COSMOS-Web_3.0_adversarial_asinh_F150W_Sep-18-2024_4class_shuffle_10_50.csv'
cosmos_f277w_fn='COSMOS-Web_3.0_adversarial_asinh_F277W_Sep-19-2024_4class_shuffle_10_50.csv'
cosmos_f444w_fn='COSMOS-Web_3.0_adversarial_asinh_F444W_Sep-19-2024_4class_shuffle_10_50.csv'


cosmos_f150w=pd.read_csv(os.path.join(data_COSMOS,cosmos_f150w_fn))
cosmos_f277w=pd.read_csv(os.path.join(data_COSMOS,cosmos_f277w_fn))
cosmos_f444w=pd.read_csv(os.path.join(data_COSMOS,cosmos_f444w_fn))


cosmos_phys_ap =Table.read(os.path.join(data_COSMOS,'COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits'))

names = [name for name in cosmos_phys_ap.colnames if len(cosmos_phys_ap[name].shape) <= 1]
cosmos_phys=cosmos_phys_ap[names].to_pandas()


merge=cosmos_f150w.merge(cosmos_f277w,how='inner',on='fullname',suffixes=(None,'_x'))
#print(len(merge))
merge2=merge.merge(cosmos_f444w,how='inner',on='fullname',suffixes=(None,'_y'))
#print(len(merge2))
cosmos_cat = merge2.merge(cosmos_phys,left_on='id',right_on='ID_SE++',how='inner')
#print(len(cosmos_cat))

filters = ['F150W','F277W','F444W']
morph=['sph','disk','irr','bd']

for f in filters:
    for m in morph:
        c = cosmos_cat.filter(regex='^'+m+'_')
        c = c.filter(regex=f+'$')
        cosmos_cat[m+'_'+f+'_mean']=c.mean(axis=1).values
        cosmos_cat[m+'_'+f+'_std']=c.std(axis=1).values


morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(cosmos_cat.sph_F277W_mean,cosmos_cat.disk_F277W_mean,cosmos_cat.irr_F277W_mean,cosmos_cat.bd_F277W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(cosmos_cat.disk_f200>0.3)]=1    
cosmos_cat['morph_flag_f277w']=np.array(morph_flag)
cosmos_cat['delta_f277']=np.array(delta_value)

morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(cosmos_cat.sph_F444W_mean,cosmos_cat.disk_F444W_mean,cosmos_cat.irr_F444W_mean,cosmos_cat.bd_F444W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(cosmos_cat.disk_f200>0.3)]=1    
cosmos_cat['morph_flag_f444w']=np.array(morph_flag)
cosmos_cat['delta_f444']=np.array(delta_value)


morph_flag=[]
delta_value = []

for sph,dk,irr,bd in zip(cosmos_cat.sph_F150W_mean,cosmos_cat.disk_F150W_mean,cosmos_cat.irr_F150W_mean,cosmos_cat.bd_F150W_mean):
    maxpos = np.argmax([sph,dk,irr,bd])
    delta = np.sort([sph,dk,irr,bd])[3]-np.sort([sph,dk,irr,bd])[2]
    morph_flag.append(maxpos)
    delta_value.append(delta)
#morph_flag=np.array(morph_flag)
#morph_flag[(cosmos_cat.disk_f200>0.3)]=1    
cosmos_cat['morph_flag_f150w']=np.array(morph_flag)
cosmos_cat['delta_f150']=np.array(delta_value)


cosmos_cat['super_compact_flag'] = (
    ((cosmos_cat['LP_zfinal'] < 1) & (cosmos_cat['RADIUS'] < (0.025 / 3600))) |
    ((cosmos_cat['LP_zfinal'] > 1) & (cosmos_cat['LP_zfinal'] < 3) & (cosmos_cat['RADIUS'] < (0.025 / 3600))) |
    ((cosmos_cat['LP_zfinal'] > 3) & (cosmos_cat['RADIUS'] < (0.025 / 3600)))
)


cosmos_cat['unc_flag'] = (
    (
        (cosmos_cat['delta_f150'] < 0.1) |
        (cosmos_cat['delta_f277'] < 0.1) |
        (cosmos_cat['delta_f444'] < 0.1)
    ) )

num_super_compact = cosmos_cat['super_compact_flag'].sum()
num_unc = cosmos_cat['unc_flag'].sum()

morph_flag = np.copy(cosmos_cat.morph_flag_f150w.values)
zbest = cosmos_cat['LP_zfinal']

morph_f356 = np.copy(cosmos_cat.morph_flag_f277w.values)
morph_f444 = np.copy(cosmos_cat.morph_flag_f444w.values)
morph_flag[(zbest>1) & (zbest<3)] = np.copy(morph_f356[(zbest>1) & (zbest<3)])
morph_flag[(zbest>3) & (zbest<6)]= np.copy(morph_f444[(zbest>3) & (zbest<6)])
super_compact_flag = cosmos_cat['super_compact_flag']
morph_flag[(super_compact_flag==1)]=-1
cosmos_cat['morph_flag']=np.copy(morph_flag)








if os.path.exists(root_COSMOS+'image_arrays/COSMOSWeb_master_v3.1.0_image_arrays_'+f+'.npz'):
        print("Loading saved array with data from filter "+f)
        data = np.load(os.path.join(root_COSMOS,'image_arrays/COSMOSWeb_master_v3.1.0_image_arrays_'+f+'.npz'),allow_pickle=True)
        # Access the saved variables
        X_JWST = data['stamps']
        fullvec = data['fullvec']
        idvec = data['idvec']
        fieldvec = data['fieldvec']
        ravec = data['ravec']
        decvec = data['decvec']


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
    

def load_imgs(tile):

    if tile in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        name_img_det        = f'/n23data2/hakins/COSMOS-Web/detection_images/detection_chi2pos_SWLW_{tile}.fits'
        sci_imas={
            'F115W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F150W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F277W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F444W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7/mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_sci.fits',
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
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
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
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'ALMA-1mm':    f'/n17data/shuntov/ALMA-CHAMPS/coadd.fits'
            }

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        name_img_det        = f'/n23data2/hakins/COSMOS-Web/detection_images/detection_chi2pos_SWLW_{tile}.fits'
        sci_imas={
            'F115W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F150W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F277W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F444W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7/mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_sci.fits',
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
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
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
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'ALMA-1mm':    f'/n17data/shuntov/ALMA-CHAMPS/coadd.fits'
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
        ver = 'v3.1.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/JAN24-{tile}_{ver}-ASC/'

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        ver = 'v3.1.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/JAN24-{tile}_{ver}-ASC/'

    # name_img_part  = path_checkimg+get_filename(path_checkimg, '', '_partition.fits')
    name_img_part = f'/n17data/shuntov/COSMOS-Web/ASSOC-files/SE-hotcold_{tile}_grouped_assoc.fits'
    model_imas = {}
    resid_imas = {}

  


    return name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate

def plot_stamps_quantiles(wl,morph,ceers_cat,data_path,nquants_z=10,nquants_mass=4,quants_stamps_z=[0,3,6,9],quants_stamps_mass=[0,1,2,3]):

    wl_low_case={
    'f115w':       f'F115W',
    'f150w':       f'F150W',
    'f277w':       f'F277W',
    'f444w':       f'F444W',
    'f770w':       f'F770W'
    }

    j=1
    k=0
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    arcsec_cut = 32*0.03
    with PdfPages(data_path+'figures/'+'morph_'+str(morph)+'_CWeb3.1_'+str(wl)+'_'+d4+'.pdf') as pdf_ceers:
        
        sel = ceers_cat.query('morph_flag_'+str(wl)+'=='+str(morph)+' and LP_zfinal>'+str(0)+' and LP_zfinal<'+str(6)+' and LP_mass_med_PDF>9')
        quant = pd.qcut(sel['LP_zfinal'].values, nquants_z,labels=False)
        print(len(quant))
        print(len(sel))
        sel['quant_z']=quant
        for qz in quants_stamps_z:
            sel_z = sel.query('quant_z=='+str(qz))
            quant_m = pd.qcut(sel_z['LP_mass_med_PDF'].values, nquants_mass,labels=False)
            sel_z['quant_mass']=quant_m
            for qm in quants_stamps_mass:
                try:
                    mcut = sel_z.query('quant_mass=='+str(qm)).sample(frac=1)
                except:
                   
                    print("nothing")
                    continue
                #j=0
                for idn,full,z,logm,ra_cent,dec_cent in zip(mcut['ID_SE++'],mcut.TILE,mcut.LP_zfinal,mcut.LP_mass_med_PDF,mcut.RA_MODEL,mcut.DEC_MODEL):
                   
                   
                    try:
                        print('making cutout')
                        name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate = load_imgs(full.decode('utf-8'))
                        stamp, w = image_make_cutout(sci_imas[wl_low_case[wl]], ra_cent, dec_cent, arcsec_cut, nameout=None, get_wcs=True)
                        #indices = np.where(idvec == idn)[0]
                        #print(indices)

                        #stamp = X_JWST[indices]
                                

                        if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                            continue
                                
                        hdu = fits.PrimaryHDU(stamp.data)
                        hdu.header.update(w.to_header())
                        hdu.writeto('tmp_ceers.fits', overwrite=True) 
                        #j+=1
                       
                                
                                

                    except:
                        print("error reading")
                        continue
                            
                            
                            
                    if j==1:
                        fig_ceers = plt.figure(1, figsize=(len(quants_stamps_mass)*10,len(quants_stamps_z)*10),clear=True)
                        ax_ceers = plt.subplot(len(quants_stamps_mass),len(quants_stamps_z),j,frameon=False)
                                
                    bb=ax_ceers.get_position()
                            
                       
                    

                        
                                
                                
                    plt.figure(1)
                    x_val = 1/len(quants_stamps_mass)-0.02
                    y_val = 1/len(quants_stamps_z)-0.02/4*len(quants_stamps_z)
                    print(x_val,y_val)
                    bounds = [0.02+x_val*np.mod((j-1),len(quants_stamps_mass)),(1-y_val-0.02/4*len(quants_stamps_z))-y_val*((j-1)//len(quants_stamps_mass)),x_val,y_val]
                    print(bounds)
                    gc = aplpy.FITSFigure('tmp_ceers.fits',figure=fig_ceers, subplot=bounds)
                    kpc_per_arcsec=cosmo.kpc_proper_per_arcmin(z)/60.
                                
                            
                    gc.axis_labels.hide()

                    gc.tick_labels.hide()
                    gc.add_scalebar(0.1 * u.arcsec)
                    #gc.scalebar.set_length(0.1/0.03 * u.pixel)
                    #gc.scalebar.set_label(str(kpc_per_arcsec*0.1))
                                
                    gc.scalebar.set_corner('bottom right')
                    scale = kpc_per_arcsec.value*0.1
                    gc.scalebar.set_label('%04.2f kpc' % scale)
                                #gc.scalebar.set_label('1 kpc')
                    gc.scalebar.set_color('black')
                    gc.scalebar.set_linestyle('solid')
                    gc.scalebar.set_linewidth(3)
                    gc.scalebar.set_font(size=30, weight='medium', \
                    stretch='normal', family='sans-serif', \
                        style='normal', variant='normal')
                    gc.show_grayscale(stretch='sqrt',invert=True)

                    ax_ceers.set_yticklabels([])
                    ax_ceers.set_xticklabels([])

                    plt.xticks([],[])
                    plt.yticks([],[])

                    plt.text(5, 55, full, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)    
                    plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                    plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                    print("z="+str(z))
                    j+=1
                    if j>25:
                        continue

                    #if j==26:
                    #    plt.tight_layout()
                    #    pdf_ceers.savefig(fig_ceers)
                    #    #pdf_candels.savefig(fig_candels)
                    #    print("saving")
                    #    j=1
                                
                                
                                
                                
                           
        plt.tight_layout()
        pdf_ceers.savefig(fig_ceers)
        
        print("final saving")

wl_vec = ['f150w','f277w','f444w']
morph_vec=[0,1,2,3]
data_out = '/n03data/huertas/COSMOS-Web/'

for wl in wl_vec:
    for morph in morph_vec:
           
        plot_stamps_quantiles(wl,morph,cosmos_cat,data_out)