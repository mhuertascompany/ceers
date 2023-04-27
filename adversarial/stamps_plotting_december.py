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
ceers_cat = pd.read_csv(data_path+"cats/CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_v051.csv")
candels_ceers = pd.read_csv(data_path+"cats/CANDELS_CEERS_match_DR05_december_ensemble_morphflag.csv")


# egs_all_wfc3_ir_f160w_030mas_v1.9_nircam1_mef.fits.gz


from matplotlib.backends.backend_pdf import PdfPages
import aplpy
from datetime import date

zbins = [0,1.5,4,6]
mbins = [9,10,10.5,11.5]



def plot_stamps_quantiles(wl,morph,ceers_cat,nir_f200_list,w,nquants_z=8,nquants_mass=4,quants_stamps_z=[0,2,4,6,8],quants_stamps_mass=[0,1,2,3]):

    j=1
    k=0
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    with PdfPages(data_path+'figures/'+'morph_'+str(morph)+'_CEERS_DR05_december_'+str(wl)+'_'+d4+'.pdf') as pdf_ceers:
        
        sel = ceers_cat.query('morph_flag_'+str(wl)+'=='+str(morph)+' and zfit_50>'+str(0)+' and zfit_50<'+str(6))
        quant = pd.qcut(sel['zfit_50'].values, nquants_z,labels=False)
        print(len(quant))
        print(len(sel))
        sel['quant_z']=quant
        for qz in quants_stamps_z:
            sel_z = sel.query('quant_z=='+str(qz))
            quant_m = pd.qcut(sel['logM_50'].values, nquants_mass,labels=False)
            sel_z['quant_mass']=quant_m
            for qm in quants_stamps_mass:
                try:
                    mcut = sel_z.query('quant_mass=='+str(qm)).sample(n=1)
                except:
                    j+=1
                    print("nothing")
                    continue
                for idn,full,ra,dec,z,logm in zip(mcut.ID_1,mcut.fullname,mcut.RA_1,mcut.DEC_1,mcut.zfit_50,mcut.logM_50):
                    read=0
                    k=0
                    while read==0:
                        if k>=10:
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
                            fig_ceers = plt.figure(1, figsize=(len(quants_stamps_mass)*10,len(quants_stamps_z)*10),clear=True)
                            ax_ceers = plt.subplot(len(quants_stamps_mass),len(quants_stamps_z),j,frameon=False)
                            
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







def plot_stamps(wl,morph,ceers_cat,nir_f200_list,w):

    j=1
    k=0
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    with PdfPages(data_path+'figures/'+'morph_'+str(morph)+'_CEERS_DR05_december_'+str(wl)+'_'+d4+'.pdf') as pdf_ceers:
        for zlow,zup in zip(zbins[:-1],zbins[1:]):
            sel = ceers_cat.query('morph_flag_'+str(wl)+'=='+str(morph)+' and zfit_50>'+str(zlow)+' and zfit_50<'+str(zup))
            #sel = ceers_cat.query('morph_flag_f200=='+str(morph)+' and delta>0.9')
            
            
            for mlow,mup in zip(mbins[:-1],mbins[1:]): 
                try:
                    mcut = sel.query("logM_50>"+str(mlow)+"and logM_50<"+str(mup)).sample(n=1)
                    #mcut = sel.sample(n=1)
                    print(mlow,mup)
                    print(zlow,zup)
                except:
                    j+=1
                    print("nothing")
                    continue
                for idn,full,ra,dec,z,logm in zip(mcut.ID_1,mcut.fullname,mcut.RA_1,mcut.DEC_1,mcut.zfit_50,mcut.logM_50):
                    read=0
                    k=0
                    while read==0:
                        if k>=10:
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


wl_vec = ['f150w','f200w','f356w','f444w']
morph_vec=[0,1,2,3]

for wl in wl_vec:
    for morph in morph_vec:
        ceers_pointings = np.arange(1,11)
        nir_f200_list=[]
        w=[]
        cats = []
        for c in ceers_pointings:
            if c==1 or c==2 or c==3 or c==6:
                nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+str(c)+"_"+wl+"_dr0.5_i2d.fits.gz")
            else:
                nir_f200 = fits.open(data_path+"images/ceers_nircam"+str(c)+"_"+wl+"_v0.51_i2d.fits.gz")    
            nir_f200_list.append(nir_f200)
            w.append(WCS(nir_f200[1].header))

        print(wl,morph)    
        plot_stamps_quantiles(wl,morph,ceers_cat,nir_f200_list,w)




