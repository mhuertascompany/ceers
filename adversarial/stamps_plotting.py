import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import pdb
import matplotlib.pyplot as plt
import h5py    
import pandas as pd

data_path = "/scratch/mhuertas/CEERS/data_release/"
ceers_cat = pd.read_csv(data_path+"cats/specz_PG_matched_SFR_mstar_z_RADEC_morphADV_200_356_400_4class.csv")
candels_ceers = pd.read_csv(data_path+"cats/CANDELS_CEERS_match_morphflag.csv")



wl = 'f200w'

ceers_pointings = ["1","2","3","6"]
nir_f200_list=[]
w=[]
cats = []
for c in ceers_pointings:
  nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+c+"_"+wl+"_dr0.5_i2d.fits.gz")
  nir_f200_list.append(nir_f200)
  w.append(WCS(nir_f200[1].header))
  cats.append(candels_ceers)

# egs_all_wfc3_ir_f160w_030mas_v1.9_nircam1_mef.fits.gz

wfc3_f160_list=[]
wf160=[]
#candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits"]
candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits"]
for c in ceers_pointings:
    wfc3_f160 = fits.open(data_path+"images/egs_all_wfc3_ir_f160w_030mas_v1.9_nircam"+c+"_mef.fits.gz")
    wfc3_f160_list.append(wfc3_f160)
    wf160.append(WCS(wfc3_f160[1].header))
    wf160[-1].sip = None


from matplotlib.backends.backend_pdf import PdfPages
import aplpy

zbins = [0,1,3,6]




mbins = [9,10,11,12]

j=1
k=0

with PdfPages(data_path+'figures/sph_CEERS_f200w.pdf') as pdf_ceers,PdfPages(data_path+'figures/sph_CANDELS_f160w.pdf') as pdf_candels:
    for zlow,zup in zip(zbins[:-1],zbins[1:]):
        sel = candels_ceers.query('(morph_flag_f200==1 or morph_flag_f200==2) and (morph_CANDELS==0 or morph_CANDELS==3) and z>'+str(zlow)+' and z<'+str(zup))
        
        
        for mlow,mup in zip(mbins[:-1],mbins[1:]): 
            try:
                mcut = sel.query("mass>"+str(mlow)+"and mass<"+str(mup)).sample(n=1)
                print(mlow,mup)
                print(zlow,zup)
            except:
                j+=1
                #print("nothing")
                continue
            for idn,ra,dec,z,logm in zip(mcut.ID_1a,mcut.RA_1,mcut.DEC_1,mcut.z,mcut.mass):
                read=0
                k=0
                while read==0:
                    if k>=4:
                        read=-1
                        continue
                    nir_f200=nir_f200_list[k]
                    w200=w[k]
                    k+=1
                    try:
                        position = SkyCoord(ra,dec,unit="deg")

                        stamp = Cutout2D(nir_f200[1].data,position,64,wcs=w200)
                        

                        if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                            continue
                        hdu = fits.PrimaryHDU(stamp.data)
                        hdu.writeto('tmp_ceers.fits', overwrite=True) 
                        print("read!")
                        print(j)
                        read=1

                    except:
                        #print("error reading")
                        continue
                     
                    
                    
                    if j==1:
                        fig_ceers = plt.figure(1, figsize=(30,30),clear=True)
                        ax_ceers = plt.subplot(3,3,j,frameon=False)
                        fig_candels = plt.figure(2,figsize=(30,30),clear=True)
                        ax_candels = plt.subplot(3,3,j,frameon=False)
                    bb=ax_ceers.get_position()
                       
                    print("here")
            

                    if read ==1:
                        nir_f160=wfc3_f160_list[k-1]
                        stamp_candels =Cutout2D(nir_f160[1].data,position,64,wcs = wf160[k-1])
                        hdu = fits.PrimaryHDU(stamp_candels.data)
                        hdu.writeto('tmp_candels.fits', overwrite=True) 
                        
                        plt.figure(1)
                        bounds = [0.02+0.16*np.mod((j-1),3),0.75+0.02-0.16*((j-1)//3),0.14,0.14]
                        gc = aplpy.FITSFigure('tmp_ceers.fits',figure=fig_ceers, subplot=bounds)
                        gc.show_grayscale(stretch='sqrt',invert=True)
                        gc.tick_labels.hide()
                        ax_ceers.set_yticklabels([])
                        ax_ceers.set_xticklabels([])

                        plt.xticks([],[])
                        plt.yticks([],[])

                        plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                        plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                        print("z="+str(z))
                        
                        plt.figure(2)
                        #bounds = [0.02+0.16*np.mod((j-1),5),0.75+0.02-0.16*((j-1)//5),0.14,0.14]
                        gc = aplpy.FITSFigure('tmp_candels.fits',figure=fig_candels, subplot=bounds)
                        gc.show_grayscale(stretch='sqrt',invert=True)
                        gc.tick_labels.hide()
                        ax_candels.set_yticklabels([])
                        ax_candels.set_xticklabels([])

                        plt.xticks([],[])
                        plt.yticks([],[])

                        plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                        plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                        print("z="+str(z))
                        
                        
                        
                        j+=1
                        print(j)
                        if j==26:
                            plt.tight_layout()
                            pdf_ceers.savefig(fig_ceers)
                            pdf_candels.savefig(fig_candels)
                            print("saving")
                            j=1
                        #k+=1
    plt.tight_layout()
    pdf_ceers.savefig(fig_ceers)
    pdf_candels.savefig(fig_candels)
    print("final saving")
