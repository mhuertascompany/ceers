import numpy as np
import emcee
import pickle
import os
import h5py

def schechter_function(logM, logphi_star, logM_star, alpha):
    M = 10**logM
    M_star = 10**logM_star
    phi_star = 10**logphi_star
    return np.log(10) * phi_star * (M / M_star)**(alpha+1) * np.exp(-M / M_star)

def double_schechter_function(logM, logM_star, alpha1, alpha2, logphi1, logphi2):
    M_over_M_star = logM - logM_star
    return np.log(10) * np.exp(-10**M_over_M_star) * 10**M_over_M_star * (10**logphi1 * 10**(M_over_M_star * alpha1) + 10**logphi2 * 10**(M_over_M_star * alpha2))

def DPL(logM, logM_star, alpha1, alpha2, logphi_star):
    term1 = 10 ** (-(logM - logM_star) * (alpha1 + 1))
    term2 = 10 ** (-(logM - logM_star) * (alpha2 + 1))
    phi_logM_star = 10**logphi_star / (term1 + term2)
    return phi_logM_star

def log_likelihood_single(theta, logM, Phi, dPhi):
    logphi_star, logM_star, alpha = theta
    model = schechter_function(logM, logphi_star, logM_star, alpha)
    return -0.5 * np.sum((Phi - model)**2 / dPhi**2)

def log_likelihood_double(theta, logM, Phi, dPhi):
    logM_star, alpha1, alpha2, logphi1, logphi2 = theta
    model = double_schechter_function(logM, logM_star, alpha1, alpha2, logphi1, logphi2)
    return -0.5 * np.sum((Phi - model)**2 / dPhi**2)

def log_likelihood_DPL(theta, logM, Phi, dPhi):
    logM_star, alpha1, alpha2, logphi_star = theta
    model = DPL(logM, logM_star, alpha1, alpha2, logphi_star)
    return -0.5 * np.sum((Phi - model)**2 / dPhi**2)

def log_prior_single(theta):
    logphi_star, logM_star, alpha = theta
    if -7 < logphi_star < 0 and 8.0 < logM_star < 12.0 and -4 < alpha < 4:
        return 0.0
    return -np.inf

def log_prior_double(theta):
    logM_star, alpha1, alpha2, logphi1, logphi2 = theta
    if -7 < logphi1 < 0 and -5 < logphi2 < 0 and 8.0 < logM_star < 12.0 and -4 < alpha1 < -1 and -1 < alpha2 < 4:
        return 0.0
    return -np.inf

def log_prior_DPL(theta):
    logM_star, alpha1, alpha2, logphi_star = theta
    if -7 < logphi_star < 0 and 8.0 < logM_star < 12.0 and -4 < alpha1 < 4 and -4 < alpha2 < 4:
        return 0.0
    return -np.inf

def log_probability_single(theta, logM, Phi, dPhi):
    lp = log_prior_single(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_single(theta, logM, Phi, dPhi)

def log_probability_double(theta, logM, Phi, dPhi):
    lp = log_prior_double(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_double(theta, logM, Phi, dPhi)

def log_probability_DPL(theta, logM, Phi, dPhi):
    lp = log_prior_DPL(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_DPL(theta, logM, Phi, dPhi)

def save_sampler_to_hdf5(sampler, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('chain', data=sampler.get_chain(), compression='gzip')
        f.create_dataset('log_prob', data=sampler.get_log_prob(), compression='gzip')

def fit_MCMC(smf_morph, path_out, filename, fit_range=(9, 12)):
    # Results dictionary
    fit_results = {}

    # Set up the MCMC parameters
    ndim_double = 5
    ndim_single = 3
    ndim_DPL = 4
    nwalkers_double = 8 * ndim_double
    nwalkers_single = 8 * ndim_single
    nwalkers_DPL = 8 * ndim_DPL
    pos_func_double = lambda g: [g + 1 * np.random.randn(ndim_double) for i in range(nwalkers_double)]
    pos_func_single = lambda g: [g + 1 * np.random.randn(ndim_single) for i in range(nwalkers_single)]
    pos_func_DPL = lambda g: [g + 1 * np.random.randn(ndim_DPL) for i in range(nwalkers_DPL)]

    # Main loop over redshift and morphology bins
    for zbin in zbins[:-1]:
        for morph in morph_class:
            try:
                if len(smf_morph[(zbin, morph, 'LogMassbin')]) > 0:
                    logM = smf_morph[(zbin, morph, 'LogMassbin')]
                    Phi = smf_morph[(zbin, morph, 'Fi')]
                    dPhi = smf_morph[(zbin, morph, 'dFi')]

                    # Filter out zero or negative values in Phi or dPhi
                    valid = (Phi > 0) & (dPhi > 0) & (logM > fit_range[0]) & (logM < fit_range[1])
                    if not np.any(valid):
                        print(f"Skipping {morph} at z={zbin} entirely, no valid data points after filtering.")
                        continue
                    logM = logM[valid]
                    Phi = Phi[valid]
                    dPhi = dPhi[valid]

                    # Double Schechter Fit
                    initial_guess_double = [10.5, -0.6, -1.7, -2, -2]
                    pos_double = pos_func_double(initial_guess_double)
                    sampler_double = emcee.EnsembleSampler(nwalkers_double, ndim_double, log_probability_double, args=(logM, Phi, dPhi))
                    sampler_double.run_mcmc(pos_double, 60000, progress=True)
                    samples_double = sampler_double.get_chain(discard=10000, thin=15, flat=True)
                    fit_results[(zbin, morph, 'double')] = {
                        'sampler': sampler_double,
                        'params_50': np.percentile(samples_double, 50, axis=0),
                        'params_16': np.percentile(samples_double, 16, axis=0),
                        'params_84': np.percentile(samples_double, 84, axis=0)
                    }
                    print(f'Done double Schechter fit for {morph} at z={zbin}')

                    # Single Schechter Fit
                    initial_guess_single = [-2, 10.5, -1.2]
                    pos_single = pos_func_single(initial_guess_single)
                    sampler_single = emcee.EnsembleSampler(nwalkers_single, ndim_single, log_probability_single, args=(logM, Phi, dPhi))
                    sampler_single.run_mcmc(pos_single, 60000, progress=True)
                    samples_single = sampler_single.get_chain(discard=10000, thin=15, flat=True)
                    fit_results[(zbin, morph, 'single')] = {
                        'sampler': sampler_single,
                        'params_50': np.percentile(samples_single, 50, axis=0),
                        'params_16': np.percentile(samples_single, 16, axis=0),
                        'params_84': np.percentile(samples_single, 84, axis=0)
                    }
                    print(f'Done single Schechter fit for {morph} at z={zbin}')

                    # DPL Fit
                    initial_guess_DPL = [10.5, -3.0, -2.0,-5.0]
                    pos_DPL = pos_func_DPL(initial_guess_DPL)
                    sampler_DPL = emcee.EnsembleSampler(nwalkers_DPL, ndim_DPL, log_probability_DPL, args=(logM, Phi, dPhi))
                    sampler_DPL.run_mcmc(pos_DPL, 60000, progress=True)
                    samples_DPL = sampler_DPL.get_chain(discard=10000, thin=15, flat=True)
                    fit_results[(zbin, morph,'DPL')] = {
                        'sampler': sampler_DPL,
                        'params_50': np.percentile(samples_DPL, 50, axis=0),
                        'params_16': np.percentile(samples_DPL, 16, axis=0),
                        'params_84': np.percentile(samples_DPL, 84, axis=0)
                    }
                    print(f'Done DPL fit for {morph} at z={zbin}')

                else:
                    print(f"Skipping {morph} at z={zbin} due to empty data.")
            except Exception as e:
                print(f"Exception occurred for {morph} at z={zbin}: {e}")
                continue 

    # Save the parameters-only results
    params_only_results = {}

    # Extract only the parameters from the original results
    for (zbin, morph, model), results in fit_results.items():
        params_only_results[(zbin, morph, model)] = {
            'params_50': results['params_50'],
            'params_16': results['params_16'],
            'params_84': results['params_84']
        }

    # Save the results to a pickle file
    #file_path = os.path.join(path_out, filename + '_fit_results_' + str(fit_range[0]) + '.pkl')
    #os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists

    # Serialize the parameters-only dictionary
    #with open(file_path, 'wb') as file:
    #    pickle.dump(params_only_results, file)

    #print(f"Parameters-only fit results stored successfully at {file_path}")

    # Save the sampler chains in HDF5 format
    samples_out_dir = os.path.join(path_out, 'samplers_12')
    os.makedirs(samples_out_dir, exist_ok=True)
    for (zbin, morph, model), results in fit_results.items():
        h5_file = os.path.join(samples_out_dir, filename + f'_{zbin}_{morph}_{model}_{fit_range[0]}_sampler.h5')
        save_sampler_to_hdf5(results['sampler'], h5_file)

    print("Sampler data saved in HDF5 format with compression.")

# Example usage with your defined morphology classes and redshift bins:
morph_class = ['irr', 'disk', 'sph', 'db','reg','comp','all']
zbins = [0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5,6.5]
#zbins = [3.5,4.5,5.5,6.5]
path_in = '/n03data/huertas/COSMOS-Web/SMF'
os.makedirs(path_in, exist_ok=True)
smf_files = ['smf_morph_3.1_allerrors_nocompact_0.07F444']

for smf_type in smf_files:
    smf_file = os.path.join(path_in, smf_type + '.pkl')
    # Load the observation dictionary
    with open(smf_file, 'rb') as file:
        smf_morph = pickle.load(file)
    fit_MCMC(smf_morph, path_in, smf_type, fit_range=(8.5, 12))
