import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
import pdb


import h5py    
import pandas as pd

import sklearn


import seaborn as sns
print(sns.__version__)
import plotly.express as px

import matplotlib as mpl
print(mpl.__version__)
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

import torch
from sbi import utils as Ut
from sbi import inference as Inference

import os
import h5py 

import copy
import optuna 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from astropy.cosmology import Planck13 as cosmo

device = ("cuda" if torch.cuda.is_available() else "cpu")



#seed = 12387
#torch.manual_seed(seed)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(seed)


data_path = "/scratch/mhuertas/CEERS/data_release/"
ceers_cat = pd.read_csv(data_path+"cats/CEERS_DR05_adversarial_asinh_3filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_v051.csv")

ceers_cat['timescale']=(10**ceers_cat.logSFRinst_50/10**ceers_cat.logM_50)/(cosmo.H(ceers_cat.zfit_50)*3.24078e-20*3.154e+7)




def sample_from_quantiles(quantile_vals, quantiles = [16,50,84], 
                          val_0 = -5, val_100 = 5, pdf_res = 1000, nsamp = 10000,
                          return_cdf = False, vb = False):
    """
    Fn to sample from an empirical PDF corresponding to input quantiles.
    Make sure val_0 and val_100 are far enough from the quantiles to avoid compressing the PDF tails
    Change nsamp for the number of samples, and pdf_res according to how finely you care about the sampling
    """

    quantiles = np.array(quantiles)
    quantile_vals = np.array(quantile_vals)

    # make an array with the quantiles
    qarr = np.zeros((len(quantiles)+2,))
    parr = np.zeros((len(quantiles)+2,))
    parr[1:-1] = quantiles
    parr[-1] = 100
    qarr[1:-1] = quantile_vals
    qarr[0] = val_0
    qarr[-1] = val_100

    # interpolate the quantile values to get the CDF
    cdf_x = np.linspace(val_0, val_100, pdf_res)
    interp_arr = pin(qarr, parr)
    cdf_y = interp_arr(cdf_x)

    # Compute the PDF if needed
    # pdf_x = cdf_x[0:-1] + np.mean(np.diff(cdf_x))/2
    # pdf_y = np.diff(cdf_y)

    # Sample from the computed CDF
    sample_qts = np.random.uniform(size=nsamp)*100
    sample_vals = cdf_x[np.array([np.argmin(np.abs(sample_qts[i] - cdf_y)) for i in range(len(sample_qts))])]

    if vb == True:
        
        plt.scatter(qarr, parr)
        plt.plot(cdf_x,cdf_y)
        plt.xlabel('x'); plt.ylabel('CDF(x)')
        plt.show()

        plt.hist(sample_vals,30, density=True)
        for i in range(len(quantiles)):
            plt.axvline(quantile_vals[i],color='k',linestyle='--',alpha=0.7)
        plt.xlabel('x');plt.ylabel('P(x)')
        plt.show()
        
        print('input quantiles and limits', qarr, parr)
        print('recovered summary quantiles', np.nanpercentile(sample_vals, quantiles))   


    if return_cdf == True:
        return cdf_x, cdf_y
    else:
        return sample_vals

# forward model
def forwardmodel_scaled(logmstar,logmstar_16,logmstar_84,alpha,beta,sfr_err_scaled): 

  logSFR = alpha*(np.random.normal(size=len(logmstar))*(logmstar_84-logmstar_16)+logmstar-10.5)+beta
  return logSFR+np.random.normal(size=len(logmstar))*(sfr_err_scaled*logSFR)


def forwardmodel(logmstar,logmstar_16,logmstar_84,alpha,beta,SFR_sort,SFR16_sort,SFR84_sort): 

  logSFR = alpha*(np.random.normal(size=len(logmstar))*(logmstar_84-logmstar_16)+logmstar-10.5)+beta
  logsfr_16 = np.interp(logSFR,SFR_sort,SFR16_sort)
  logsfr_84 = np.interp(logSFR,SFR_sort,SFR84_sort)
  return logSFR+np.random.normal(size=len(logmstar))*(logsfr_84-logsfr_16)



def create_sims(ceers_cat,nsims,zbin,timescale):
    sel = ceers_cat.query("timescale>0.33 and zfit_50>"+str(zbin[0])+"and zfit_50<"+str(zbin[1])+" and logM_50>9 and logM_50<10.3 and logM_16.notna() and logM_84.notna() and logSFR100_16.notna() and logSFR100_84.notna()")
    mass=sel['logM_50'].values
    mass_16 = sel['logM_16'].values
    mass_84 = sel['logM_84'].values
    sfr_16 = sel['logSFR100_16'].values
    sfr_84 = sel['logSFR100_84'].values
    sfr = sel['logSFR100_50'].values
    sfr_err_scaled = (sfr_84-sfr_16)/sel['logSFR100_50']
    #SFR=ceers_cat.logSFR100_50.values
    #SFR16 = ceers_cat.logSFR100_16.values
    nd = np.argsort(sfr, axis=0) 



    alpha_range = [-.3,1.3]
    beta_range=[-1,2]
    #sigma_range=[0,0.5]

    
    alpha = np.random.uniform(low=alpha_range[0],high=alpha_range[1],size=nsims)
    beta = np.random.uniform(low=beta_range[0],high=beta_range[1],size=nsims)
    #sigma = np.random.uniform(low=beta_range[0],high=beta_range[1],size=nsims)

    thetas=np.zeros((nsims,2))
    thetas[:,0]=alpha
    thetas[:,1]=beta
    #thetas[:,2]=sigma

    return thetas, np.array([forwardmodel(mass,mass_16,mass_84,tt[0], tt[1],sfr[nd],sfr_16[nd],sfr_84[nd]) for tt in thetas])




def Objective(trial):
    ''' objective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)

    lr  = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 

    p_drop = trial.suggest_float("p_drop", p_drop_min, p_drop_max)
    clip_max = trial.suggest_float("clip_max_norm", clip_max_min, clip_max_max) 

    neural_posterior = Ut.posterior_nn('maf', 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            dropout_probability=p_drop, 
            use_batch_norm=True)
    # initialize
    
    
    npe = neural_posterior(torch.from_numpy(thetas_train.astype(np.float32)), torch.from_numpy(sfrs_train.astype(np.float32)))
    npe.to(device)

    # train NDE 
    optimizer = optim.Adam(list(npe.parameters()), lr=lr)
    # set up scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=1000)

    best_valid_loss, best_epoch = np.inf, 0
    train_losses, valid_losses = [], []
    for epoch in range(1000):

        npe.train()
        print(epoch)
        train_loss = 0.
        for batch in train_loader:
            optimizer.zero_grad()
            
            theta, sfrs = (batch[0].to(device), batch[1].to(device))
    
            logprobs = npe.log_prob(theta, context=sfrs) 
            
            loss = -torch.sum(logprobs) 
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        with torch.no_grad():
            valid_loss = 0.  
        
            for batch in valid_loader:
                theta, bovera = (batch[0].to(device), batch[1].to(device))

                logprobs = npe.log_prob(theta, context=bovera) 
            
                loss = -torch.sum(logprobs) 

                valid_loss += loss.item()
            valid_loss /= len(valid_loader.dataset)

        if (epoch % 10 == 0) and (epoch != 0):
            print('Epoch %i Training Loss %.2e Validation Loss %.2e' % (epoch, train_loss, valid_loss))
            fqphi   = os.path.join(output_dir, study_name, '%s.%i.%i.pt' % (study_name, trial.number,epoch))
            torch.save(best_npe, fqphi)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_npe = copy.deepcopy(npe)

        if epoch > best_epoch + 20: break
        scheduler.step()

    # save trained NPE  
    #qphi = best_npe.build_posterior()
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(best_npe, fqphi)





output_dir = '/scratch/mhuertas/CEERS/SFMS_fits/'
nsims=100000
timescale=0.33
n_blocks_min, n_blocks_max = 2, 10
n_transf_min, n_transf_max = 2, 10
n_hidden_min, n_hidden_max = 64, 512
n_lr_min, n_lr_max = 1e-4, 1e-2 
p_drop_min, p_drop_max = 0., 1.
clip_max_min, clip_max_max = 1., 5.



zbins = [0,1,3,6]

for zlow,zup in zip(zbins[:-1],zbins[1:]):
  thetas_train,sfrs_train = create_sims(ceers_cat,nsims,[zlow,zup],timescale)
  thetas_valid,sfrs_valid = create_sims(ceers_cat,int(nsims*0.2),[zlow,zup],timescale)


  train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(thetas_train.astype(np.float32)),
            torch.from_numpy(sfrs_train.astype(np.float32))),
        batch_size=128, shuffle=True)


  valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(thetas_valid.astype(np.float32)),
            torch.from_numpy(sfrs_valid.astype(np.float32))),
        batch_size=128, shuffle=True)









  # Optuna Parameters
  n_trials    = 5
  study_name  = 'SFMS.powerlaw.noclip.zsteve.interp.timescale.'+str(zlow)+'.'+str(zup)
  n_jobs     = 1

  if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
  storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
  n_startup_trials = 20


  sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
  study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)
  study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)








