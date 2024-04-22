import numpy as np
import emcee
import pickle
import os
import h5py

morph_class=['sph','disk','irr','db','early','all','reg','late']
zbins =  [0.2,0.5,0.8,1.1,1.5,2,2.5,3,3.5,4.5,5.5]

def double_schechter_function(logM, logM_star,alpha1,alpha2,logphi1,logphi2):
    M_over_M_star = logM - logM_star
    
    return np.log(10)*np.exp(-10**M_over_M_star)*10**M_over_M_star*(10**logphi1*10**(M_over_M_star*alpha1)+10**logphi2*10**(M_over_M_star*alpha2))

def log_likelihood(theta, logM, Phi, dPhi):
    logM_star, alpha1, alpha2, logphi1, logphi2 = theta
    model = double_schechter_function(logM, logM_star, alpha1, alpha2, logphi1, logphi2)
    return -0.5 * np.sum((Phi - model)**2 / dPhi**2)

def log_prior(theta):
    logM_star, alpha1, alpha2, logphi1, logphi2 = theta
    if -5 < logphi1 < 0 and -5 < logphi2 < 0 and 8.0 < logM_star < 12.0 and -4 < alpha1 < -1 and -1 < alpha2 < 4:
        return 0.0
    return -np.inf

def log_probability(theta, logM, Phi, dPhi):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, logM, Phi, dPhi)

def save_sampler_to_hdf5(sampler, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('chain', data=sampler.get_chain(), compression='gzip')
        f.create_dataset('log_prob', data=sampler.get_log_prob(), compression='gzip')


def double_schechter_MCMC(smf_morph,path_out,filename,fit_range=(9,11.5)):
    
    
    
    # Results dictionary
    fit_results = {}

    # Set up the MCMC parameters
    nwalkers, ndim = 500, 5
    pos_func = lambda g: [g + 1*np.random.randn(ndim) for i in range(nwalkers)]

    # Main loop over redshift and morphology bins
    for zbin in zbins[:-1]:
        for morph in morph_class:
            #key = (zbin, morph)
            if len(smf_morph[(zbin,morph,'LogMassbin')]) > 0:
                logM = smf_morph[(zbin,morph,'LogMassbin')]
                Phi = smf_morph[(zbin,morph,'Fi')]
                dPhi = smf_morph[(zbin,morph,'dFi')]

                # Filter out zero or negative values in Phi or dPhi
                valid = (Phi > 0) & (dPhi > 0) &(logM>fit_range[0] &(logM<fit_range[1]))
                if not np.any(valid):
                    print(f"Skipping {morph} at z={zbin} entirely, no valid data points after filtering.")
                    continue
                logM = logM[valid]
                Phi = Phi[valid]
                dPhi = dPhi[valid]

                initial_guess = [10.5, -0.6, -1.7, -2, -2]
                pos = pos_func(initial_guess)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(logM, Phi, dPhi))
                sampler.run_mcmc(pos, 10000, progress=True)

                # Store the sampler chain and best-fit parameters
                samples = sampler.get_chain(discard=2000, thin=15, flat=True)
                logM_star_50, alpha1_50, alpha2_50, logphi1_50, logphi2_50 = np.percentile(samples, 50, axis=0)
                logM_star_16, alpha1_16, alpha2_16, logphi1_16, logphi2_16 = np.percentile(samples, 16, axis=0)
                logM_star_84, alpha1_84, alpha2_84, logphi1_84, logphi2_84 = np.percentile(samples, 84, axis=0)
                fit_results[(zbin,morph)] = {'sampler': sampler, 'params_50': (logM_star_50, alpha1_50, alpha2_50, logphi1_50, logphi2_50),'params_16': (logM_star_16, alpha1_16, alpha2_16, logphi1_16, logphi2_16),'params_84': (logM_star_84, alpha1_84, alpha2_84, logphi1_84, logphi2_84)}
                print('done fit ',zbin,morph)
            else:
                print(f"Skipping {morph} at z={zbin} due to empty data.")


    # Assume fit_results is loaded or available from prior code
    params_only_results = {}

    # Extract only the parameters from the original results
    for (zbin, morph), results in fit_results.items():
        params_only_results[(zbin, morph)] = {
            'params_50': results['params_50'],
            'params_16': results['params_16'],
            'params_84': results['params_84']
        }

    # Save the results to a pickle file
    file_path = os.path.join(path_out,filename+'.pkl')
    #'/Users/marchuertascompany/Documents/data/COSMOS-Web/SMF/smf_fit_results.pkl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists

    # Serialize the parameters-only dictionary
    with open(file_path, 'wb') as file:
        pickle.dump(params_only_results, file)

    print(f"Parameters-only fit results stored successfully at {file_path}")



   
    samples_out_dir=os.path.join(path_out,'samplers')
    # Create the directory if it does not exist
    os.makedirs(samples_out_dir, exist_ok=True)
    for (zbin, morph), results in fit_results.items():
        h5=os.path.join(samples_out_dir,filename+f'_{zbin}_{morph}_sampler.h5')
        #filename = f'/Users/marchuertascompany/Documents/data/COSMOS-Web/SMF/{zbin}_{morph}_sampler.h5'
        save_sampler_to_hdf5(results['sampler'], h5)

    print("Sampler data saved in HDF5 format with compression.")


path_in='/n03data/huertas/COSMOS-Web/SMF'
os.makedirs(path_in, exist_ok=True)
smf_files=['smf_morph_Q','smf_morph_SF'] 

for smf_type in smf_files:
    smf_file=os.path.join(path_in,smf_type+'.pkl')
    # Load the observation dictionary
    with open(smf_file, 'rb') as file:
        smf_morph = pickle.load(file)
    double_schechter_MCMC(smf_morph,path_in,smf_type,fit_range=(9,11.5))

