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

import aplpy
from matplotlib.backends.backend_pdf import PdfPages

data_path = "/scratch/mhuertas/CEERS/"
ceers_cat = pd.read_csv(data_path+"specz_PG_matched_SFR_mstar_z_RADEC_morphADV_200_356_444_4class.csv")
candels_cat = pd.read_csv(data_path+"CANDELS_morphology.csv")


morph_flag=[]

for sph,dk,irr in zip(ceers_cat.sph_356,ceers_cat.disk_356,ceers_cat.irr_356):
    maxpos = np.argmax([sph,dk,irr])
    morph_flag.append(maxpos)
    
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f356>0.3)]=1
#morph_flag[(ceers_cat.irr_f356>0.3) & (ceers_cat.sph_f356>0.3)]=3
ceers_cat['morph_flag_f356']=np.array(morph_flag)

morph_flag=[]


for sph,dk,irr,bd in zip(ceers_cat.sph_200,ceers_cat.disk_200,ceers_cat.irr_200,1-(ceers_cat.sph_200+ceers_cat.disk_200+ceers_cat.irr_200)):
    maxpos = np.argmax([sph,dk,irr,bd])
    #pdb.set_trace()
    morph_flag.append(maxpos)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f200']=np.array(morph_flag)


morph_flag=[]

for sph,dk,irr in zip(ceers_cat.sph_444,ceers_cat.disk_444,ceers_cat.irr_444):
    maxpos = np.argmax([sph,dk,irr])
    morph_flag.append(maxpos)
#morph_flag=np.array(morph_flag)
#morph_flag[(ceers_cat.disk_f200>0.3)]=1    
ceers_cat['morph_flag_f444']=np.array(morph_flag)

#ceers_cat.to_csv(data_path+"CEERS_v005_adamatchmorph_200_irr02_barro_PG_100_tau09_morphflag.csv")



wfc3_f160_list=[]
wf160=[]
candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits"]
for c in candels_images:
    wfc3_f160 = fits.open(data_path+c)
    wfc3_f160_list.append(wfc3_f160)
    wf160.append(WCS(wfc3_f160[0].header))
    wf160[-1].sip = None







def plot_stamps(filter='f200w'):

    #filter ='f200w'


    ceers_pointings = ["1","2","3","6"]
    #ceers_pointings = ["1","3","6"]
    cat_ceers =   pd.read_csv(data_path+"specz_PG_matched_SFR_mstar_z_RADEC.csv")
    #ceers_pointings = ["2"]
    nir_f200_list=[]
    w=[]
    cats = []
    for c in ceers_pointings:
        nir_f200 = fits.open(data_path+"ceers_nircam"+c+"_"+filter+"_v0.1_mbkgsub1.fits")
        nir_f200_list.append(nir_f200)
        w.append(WCS(nir_f200['SCI'].header))
  #pdb.set_trace()  
  
        cats.append(cat_ceers.query("pointing=='nircam"+c+"'"))   #CEERS_NIRCam6_v0.07.4_photom.fits  

    zlow=1
    zbin=0.5
    zmax=zlow+zbin




    mbins = [9.5,10,10.5,11,11.5]

    j=1
    k=0

    with PdfPages('sph_CEERS_'+filter+'.pdf') as pdf:
        while zmax<5:
            sel = ceers_cat.query('(morph_flag_f200==0) and z>'+str(zlow)+' and z<'+str(zmax))
            
            zlow+=zbin
            zmax+=zbin
            
            for mlow,mup in zip(mbins[:-1],mbins[1:]): 
                try:
                    mcut = sel.query("mass>"+str(mlow)+"and mass<"+str(mup)).sample(n=1)
                    print(mlow,mup)
                    print(zlow,zmax)
                except:
                    j+=1
                    #print("nothing")
                    continue
                for idn,ra,dec,z,logm in zip(mcut.ID_CEERS,mcut.RA_1,mcut.DEC_1,mcut.z,mcut.mass):
                    read=0
                    k=0
                    print(k)
                    while read==0:
                        nir_f200=nir_f200_list[k]
                        w200=w[k]
                        k+=1
                        try:
                            position = SkyCoord(ra,dec,unit="deg")

                            stamp = Cutout2D(nir_f200[1].data,position,64,wcs=w200)

                            if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                                continue
                            hdu = fits.PrimaryHDU(stamp.data)
                            hdu.writeto('tmp.fits', overwrite=True) 
                            print("read!")
                            print(j)
                            read=1

                        except:
                            #print("error reading")
                            continue
                            
                        if j==1:
                            fig = plt.figure(figsize=(50,50))
                            ax = plt.subplot(5,5,j,frameon=False)
                        bb=ax.get_position()
                        
                        print("here")
                

                        if read ==1:    
                            bounds = [0.02+0.16*np.mod((j-1),5),0.75+0.02-0.16*((j-1)//5),0.14,0.14]
                            gc = aplpy.FITSFigure('tmp.fits',figure=fig, subplot=bounds)
                            gc.show_grayscale(stretch='sqrt',invert=True)
                            gc.tick_labels.hide()
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])

                            plt.xticks([],[])
                            plt.yticks([],[])

                            plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            print("z="+str(z))
                            j+=1
                            print(j)
                            if j==26:
                                plt.tight_layout()
                                pdf.savefig()
                                print("saving")
                                j=1
                            #k+=1
        plt.tight_layout()
        pdf.savefig()
        print("final saving")


    zlow=1
    zbin=0.5
    zmax=zlow+zbin

    mbins = [9.5,10,10.5,11,11.5]

    j=1
    k=0


    with PdfPages('disk_CEERS_'+filter+'.pdf') as pdf:
        while zmax<5:
            sel = ceers_cat.query('(morph_flag_f200==1) and z>'+str(zlow)+' and z<'+str(zmax))
            zlow+=zbin
            zmax+=zbin
            
            for mlow,mup in zip(mbins[:-1],mbins[1:]): 
                try:
                    mcut = sel.query("mass>"+str(mlow)+"and mass<"+str(mup)).sample(n=1)
                    print(mlow,mup)
                    print(zlow,zmax)
                except:
                    j+=1
                    #print("nothing")
                    continue
                for idn,ra,dec,z,logm in zip(mcut.ID_CEERS,mcut.RA_1,mcut.DEC_1,mcut.z,mcut.mass):
                    read=0
                    k=0
                    print(k)
                    while read==0:
                        nir_f200=nir_f200_list[k]
                        w200=w[k]
                        k+=1
                        try:
                            position = SkyCoord(ra,dec,unit="deg")

                            stamp = Cutout2D(nir_f200[1].data,position,64,wcs=w200)

                            if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                                continue
                            hdu = fits.PrimaryHDU(stamp.data)
                            hdu.writeto('tmp.fits', overwrite=True) 
                            print("read!")
                            print(j)
                            read=1

                        except:
                            #print("error reading")
                            continue
                            
                        if j==1:
                            fig = plt.figure(figsize=(50,50))
                            ax = plt.subplot(5,5,j,frameon=False)
                        bb=ax.get_position()
                        
                        print("here")
                

                        if read ==1:    
                            bounds = [0.02+0.16*np.mod((j-1),5),0.75+0.02-0.16*((j-1)//5),0.14,0.14]
                            gc = aplpy.FITSFigure('tmp.fits',figure=fig, subplot=bounds)
                            gc.show_grayscale(stretch='sqrt',invert=True)
                            gc.tick_labels.hide()
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])

                            plt.xticks([],[])
                            plt.yticks([],[])

                            plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            print("z="+str(z))
                            j+=1
                            print(j)
                            if j==26:
                                plt.tight_layout()
                                pdf.savefig()
                                print("saving")
                                j=1
                            #k+=1
        plt.tight_layout()
        pdf.savefig()
        print("final saving")

    zlow=1
    zbin=0.5
    zmax=zlow+zbin

    mbins = [9.5,10,10.5,11,11.5]

    j=1
    k=0   


    with PdfPages('irr_CEERS_'+filter+'.pdf') as pdf:
        while zmax<5:
            sel = ceers_cat.query('(morph_flag_f200==2) and z>'+str(zlow)+' and z<'+str(zmax))
            zlow+=zbin
            zmax+=zbin
            
            for mlow,mup in zip(mbins[:-1],mbins[1:]): 
                try:
                    mcut = sel.query("mass>"+str(mlow)+"and mass<"+str(mup)).sample(n=1)
                    print(mlow,mup)
                    print(zlow,zmax)
                except:
                    j+=1
                    #print("nothing")
                    continue
                for idn,ra,dec,z,logm in zip(mcut.ID_CEERS,mcut.RA_1,mcut.DEC_1,mcut.z,mcut.mass):
                    read=0
                    k=0
                    print(k)
                    while read==0:
                        #pdb.set_trace()
                        nir_f200=nir_f200_list[k]
                        w200=w[k]
                        k+=1
                        try:
                            position = SkyCoord(ra,dec,unit="deg")

                            stamp = Cutout2D(nir_f200[1].data,position,64,wcs=w200)

                            if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                                continue
                            hdu = fits.PrimaryHDU(stamp.data)
                            hdu.writeto('tmp.fits', overwrite=True) 
                            print("read!")
                            print(j)
                            read=1

                        except:
                            #print("error reading")
                            continue
                            
                        if j==1:
                            fig = plt.figure(figsize=(50,50))
                            ax = plt.subplot(5,5,j,frameon=False)
                        bb=ax.get_position()
                        
                        print("here")
                

                        if read ==1:    
                            bounds = [0.02+0.16*np.mod((j-1),5),0.75+0.02-0.16*((j-1)//5),0.14,0.14]
                            gc = aplpy.FITSFigure('tmp.fits',figure=fig, subplot=bounds)
                            gc.show_grayscale(stretch='sqrt',invert=True)
                            gc.tick_labels.hide()
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])

                            plt.xticks([],[])
                            plt.yticks([],[])

                            plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            print("z="+str(z))
                            j+=1
                            print(j)
                            if j==26:
                                plt.tight_layout()
                                pdf.savefig()
                                print("saving")
                                j=1
                            #k+=1
        plt.tight_layout()
        pdf.savefig()
        print("final saving")    


    zlow=1
    zbin=0.5
    zmax=zlow+zbin

    mbins = [9.5,10,10.5,11,11.5]

    j=1
    k=0   


    with PdfPages('bd_CEERS_'+filter+'.pdf') as pdf:
        while zmax<5:
            sel = ceers_cat.query('(morph_flag_f200==3) and z>'+str(zlow)+' and z<'+str(zmax))
            zlow+=zbin
            zmax+=zbin
            
            for mlow,mup in zip(mbins[:-1],mbins[1:]): 
                try:
                    mcut = sel.query("mass>"+str(mlow)+"and mass<"+str(mup)).sample(n=1)
                    print(mlow,mup)
                    print(zlow,zmax)
                except:
                    j+=1
                    #print("nothing")
                    continue
                for idn,ra,dec,z,logm in zip(mcut.ID_CEERS,mcut.RA_1,mcut.DEC_1,mcut.z,mcut.mass):
                    read=0
                    k=0
                    print(k)
                    while read==0:
                        #pdb.set_trace()
                        nir_f200=nir_f200_list[k]
                        w200=w[k]
                        k+=1
                        try:
                            position = SkyCoord(ra,dec,unit="deg")

                            stamp = Cutout2D(nir_f200[1].data,position,64,wcs=w200)

                            if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                                continue
                            hdu = fits.PrimaryHDU(stamp.data)
                            hdu.writeto('tmp.fits', overwrite=True) 
                            print("read!")
                            print(j)
                            read=1

                        except:
                            #print("error reading")
                            continue
                            
                        if j==1:
                            fig = plt.figure(figsize=(50,50))
                            ax = plt.subplot(5,5,j,frameon=False)
                        bb=ax.get_position()
                        
                        print("here")
                

                        if read ==1:    
                            bounds = [0.02+0.16*np.mod((j-1),5),0.75+0.02-0.16*((j-1)//5),0.14,0.14]
                            gc = aplpy.FITSFigure('tmp.fits',figure=fig, subplot=bounds)
                            gc.show_grayscale(stretch='sqrt',invert=True)
                            gc.tick_labels.hide()
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])

                            plt.xticks([],[])
                            plt.yticks([],[])

                            plt.text(5, 5, '$\log M_*=$'+'%04.2f' % logm, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            plt.text(5, 15, '$z=$'+'%04.2f' % z, bbox={'facecolor': 'white', 'pad': 10},fontsize=50)
                            print("z="+str(z))
                            j+=1
                            print(j)
                            if j==26:
                                plt.tight_layout()
                                pdf.savefig()
                                print("saving")
                                j=1
                            #k+=1
        plt.tight_layout()
        pdf.savefig()
        print("final saving")    


plot_stamps('f200w')
plot_stamps('f356w')
plot_stamps('f356w')