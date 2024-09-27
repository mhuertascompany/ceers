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
        X_JWST = data['stamps'].values
        fullvec = data['fullvec'].values
        idvec = data['idvec'].values
        fieldvec = data['fieldvec'].values
        ravec = data['ravec'].values
        decvec = data['decvec'].values



def plot_stamps_quantiles(wl,morph,ceers_cat,data_path,nquants_z=10,nquants_mass=4,quants_stamps_z=[0,3,6,9],quants_stamps_mass=[0,1,2,3]):

    j=1
    k=0
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
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
                for idn,full,z,logm in zip(mcut['ID_SE++'],mcut.fullname,mcut.LP_zfinal,mcut.LP_mass_med_PDF):
                   
                   
                    if True:
                        indices = np.where(idvec == idn)[0]
                        print(indices)

                        stamp = X_JWST[indices]
                                

                        if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                            continue
                                
                        hdu = fits.PrimaryHDU(stamp.data)
                        hdu.header.update(stamp.wcs.to_header())
                        hdu.writeto('tmp_ceers.fits', overwrite=True) 
                        #j+=1
                       
                                
                                

                    else:
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
                                
                                
                                
                                
                           
        plt.tight_layout()
        pdf_ceers.savefig(fig_ceers)
        
        print("final saving")

wl_vec = ['f150w','f200w','f356w','f444w']
morph_vec=[0,1,2,3]
data_out = '/n03data/huertas/COSMOS-Web/'

for wl in wl_vec:
    for morph in morph_vec:
           
        plot_stamps_quantiles(wl,morph,cosmos_cat,data_out)