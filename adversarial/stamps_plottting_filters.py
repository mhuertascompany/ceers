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
from astropy import units as u
from astropy.cosmology import Planck13 as cosmo


data_path = "/scratch/mhuertas/CEERS/data_release/"
ceers_cat = pd.read_csv(data_path+"cats/CEERS_v01_specz_PG_matched_SFR_mstar_z_RADEC_morphADV_200_356_400_4class_morphflag.csv")
candels_ceers = pd.read_csv(data_path+"cats/CANDELS_CEERS_match_morphflag.csv")







# egs_all_wfc3_ir_f160w_030mas_v1.9_nircam1_mef.fits.gz


from matplotlib.backends.backend_pdf import PdfPages
import aplpy

zbins = [0,1,3,6]




mbins = [9,10,10.5,11.5]


def plot_stamps(wl,morph,ceers_cat,nir_f200_list,w):

    j=1
    k=0

    with PdfPages(data_path+'figures/'+'morph_'+str(morph)+'_CEERS_'+str(wl)+'.pdf') as pdf_ceers:
        for zlow,zup in zip(zbins[:-1],zbins[1:]):
            sel = ceers_cat.query('morph_flag_f200=='+str(morph)+' and z>'+str(zlow)+' and z<'+str(zup))
            
            
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
                            #pdb.set_trace()
                            hdu = fits.PrimaryHDU(stamp.data)
                            hdu.header.update(stamp.wcs.to_header())
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
                            
                        bb=ax_ceers.get_position()
                        
                        print("here")
                

                        if read ==1:
                            
                            
                            plt.figure(1)
                            bounds = [0.02+0.32*np.mod((j-1),3),0.64+0.02-0.32*((j-1)//3),0.32,0.32]
                            gc = aplpy.FITSFigure('tmp_ceers.fits',figure=fig_ceers, subplot=bounds)
                            kpc_per_arcsec=cosmo.kpc_proper_per_arcmin(z)/60.
                            
                           
                            gc.axis_labels.hide()

                            gc.tick_labels.hide()
                            gc.add_scalebar(0.1 * u.arcsec)
                            #gc.scalebar.set_length(0.1/0.03 * u.pixel)
                            #gc.scalebar.set_label(str(kpc_per_arcsec*0.1))
                            
                            gc.scalebar.set_corner('bottom right')
                            gc.scalebar.set_label(str(kpc_per_arcsec*0.1)+" kpc")
                            #gc.scalebar.set_label('1 kpc')
                            gc.scalebar.set_color('black')
                            gc.scalebar.set_linestyle('solid')
                            gc.scalebar.set_linewidth(3)
                            gc.scalebar.set_font(size='medium', weight='medium', \
                      stretch='normal', family='sans-serif', \
                      style='normal', variant='normal')
                            gc.show_grayscale(stretch='sqrt',invert=True)

                            ax_ceers.set_yticklabels([])
                            ax_ceers.set_xticklabels([])

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
                                
                                print("saving")
                                j=1
                            #k+=1
        plt.tight_layout()
        pdf_ceers.savefig(fig_ceers)
        
        print("final saving")


wl_vec = ['f200w','f356w','f444w']
morph_vec=[0,1,2,3]

for wl in wl_vec:
    for morph in morph_vec:
        ceers_pointings = ["1","2","3","6"]
        nir_f200_list=[]
        w=[]
        cats = []
        for c in ceers_pointings:
            nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+c+"_"+wl+"_dr0.5_i2d.fits.gz")
            nir_f200_list.append(nir_f200)
            w.append(WCS(nir_f200[1].header))

        print(wl,morph)    
        plot_stamps(wl,morph,ceers_cat,nir_f200_list,w)