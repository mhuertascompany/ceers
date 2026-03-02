import numpy as np
import pandas as pd

import h5py  

import pdb
 
import torch
from typing import Optional, Tuple, Union
from torch import Tensor

import pickle
from operator import itemgetter

import os
import sys
sys.path.append('/scratch/lmarrero-ext/florah/src')

from florah.models.rnn_model.rnn_generator import DataModule 




# --- PART 1 ---- Functions



def build_roots_optimized(cosmos_cat, mass_bin, nsamples=100, zbins=[0, 0.5, 1], sample_fraction=0.3):
    """
    Optimized version of build_roots.
    Generates initial pairs of galaxies (root -> progenitor candidate), to initialize trees
    """
    
    # ---------------------------------------------------------
    # 1. Prepare data pools
    # ---------------------------------------------------------
    
    # Pool A: Roots (galaxies w redshift between zbins[0] y zbins[1], and mass in mass_bin)
    mask_roots = (
        (cosmos_cat['zpdf_med'] > zbins[0]) & 
        (cosmos_cat['zpdf_med'] < zbins[1]) & 
        (cosmos_cat['mass_CIGALE'] > mass_bin[0]) & 
        (cosmos_cat['mass_CIGALE'] < mass_bin[1])
    )
    pool_roots = cosmos_cat[mask_roots].copy()
    
    # Pool B: Choose progenitor candidates (galaxies w redshift between zbins[1] y zbins[2])
    # Note: We will filter mass later, because it will depend on each root's mass.
    mask_candidates = (
        (cosmos_cat['zpdf_med'] > zbins[1]) & 
        (cosmos_cat['zpdf_med'] < zbins[2])
    )
    pool_candidates = cosmos_cat[mask_candidates].copy()
    
    # If unable to find any galaxy (weird), return empty
    if pool_roots.empty or pool_candidates.empty:
        print(f"En build_roots_optimized con mass_bin {mass_bin} - Advertencia: No se encontraron galaxias suficientes en los rangos de Z o Masa.")
        empty_out = {'x': [], 't': [], 'ra': [], 'dec': [], 'sersic': [], 'bovert': [], 'morphology': [], 'id': []}
        return empty_out, 0, []

    # ---------------------------------------------------------
    # 2. Root selection (w nsamples)
    # ---------------------------------------------------------
    if len(pool_roots) < nsamples:
        # If theres less roots in pool than nsamples, will take all but shuffled
        pool_roots = pool_roots.sample(frac=1)
    else:
        # If theres more, we will take only a number nsamples of roots
        pool_roots = pool_roots.sample(n=nsamples)
        
    n_roots = len(pool_roots)
    
    # ---------------------------------------------------------
    # 3. EXTRACCIÓN A NUMPY (VELOCIDAD MÁXIMA)
    # ---------------------------------------------------------
    # Convert DataFrames to dictionary of arrays for easy access
    # Roots
    r_mass = pool_roots['mass_CIGALE'].values
    r_sfr  = pool_roots['sfr_CIGALE'].values
    r_rad  = pool_roots['log_radius_kpc'].values
    r_z    = pool_roots['zpdf_med'].values
    r_ra   = pool_roots['ra'].values
    r_dec  = pool_roots['dec'].values
    r_ser  = pool_roots['sersic'].values
    r_bov  = pool_roots['bovert'].values
    r_mor  = pool_roots['morphology'].values
    r_id   = pool_roots['id'].values

    # Candidates
    c_mass = pool_candidates['mass_CIGALE'].values
    c_sfr  = pool_candidates['sfr_CIGALE'].values
    c_rad  = pool_candidates['log_radius_kpc'].values
    c_z    = pool_candidates['zpdf_med'].values
    c_ra   = pool_candidates['ra'].values
    c_dec  = pool_candidates['dec'].values
    c_sersic = pool_candidates['sersic'].values
    c_bovert = pool_candidates['bovert'].values
    c_morph = pool_candidates['morphology'].values
    c_id   = pool_candidates['id'].values

    # Store results in a list
    out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id']}
    len_sel2_vec = []
    
    total_pairs = 0

    # ---------------------------------------------------------
    # 4. Main loop (iterate over each root)
    # ---------------------------------------------------------
    for i in range(n_roots):
        
        # Mass of current root
        current_mass = r_mass[i]
        
        # FILTER: Find candidates that fulfill mass condition
        # Condition: mass root - 1.5 < mass candidato < mass_root + 0.5
        mask_matches = (c_mass > (current_mass - 1.5)) & (c_mass < (current_mass + 0.5))
        
        # Index of those candidates that fulfill condition
        match_indices = np.where(mask_matches)[0]
        n_matches = len(match_indices)
        
        # FILTER: Sampling a fraction of those candidates (by default sample_fraction=0.3)
        if n_matches > 0:
            n_sample = int(n_matches * sample_fraction)
            
            if n_sample > 0:
                chosen_indices = np.random.choice(match_indices, size=n_sample, replace=False)
                
                # Create pairs (root -> candidate)
                # Iterate over the chosen candidates to build pairs
                for idx in chosen_indices:
                    # ROOTS
                    row0_x = [r_mass[i], r_rad[i], r_sfr[i]]
                    row0_t = [1.0 / (1.0 + r_z[i])]
                    row0_ra = [r_ra[i]]
                    row0_dec = [r_dec[i]]
                    row0_ser = [r_ser[i]]
                    row0_bov = [r_bov[i]]
                    row0_mor = [r_mor[i]]
                    row0_id = [r_id[i]]
                    
                    # Candidates
                    row1_x = [c_mass[idx], c_rad[idx], c_sfr[idx]]
                    row1_t = [1.0 / (1.0 + c_z[idx])]
                    row1_ra = [c_ra[idx]]
                    row1_dec = [c_dec[idx]]
                    row1_ser = [c_sersic[idx]]
                    row1_bov = [c_bovert[idx]]
                    row1_mor = [c_morph[idx]]
                    row1_id = [c_id[idx]]
                    
                    # vstack to create arrays of shape (2, N)
                    out['x'].append(np.array([row0_x, row1_x], dtype=np.float32))
                    out['t'].append(np.array([row0_t, row1_t], dtype=np.float32)) # Shape (2,1)
                    out['ra'].append(np.array([row0_ra, row1_ra], dtype=np.float32))
                    out['dec'].append(np.array([row0_dec, row1_dec], dtype=np.float32))
                    out['sersic'].append(np.array([row0_ser, row1_ser], dtype=np.float32))
                    out['bovert'].append(np.array([row0_bov, row1_bov], dtype=np.float32))
                    out['morphology'].append(np.array([row0_mor, row1_mor], dtype=np.float32))
                    out['id'].append(np.array([row0_id, row1_id], dtype=np.float32))
                    
                    total_pairs += 1
            
            len_sel2_vec.append(n_sample) # Store how many galaxies were sampled in each iteration
        else:
            len_sel2_vec.append(0)

    print(f'LLegados a este punto, build_roots_optimized ha funcionado bien para mass_bin {mass_bin}')
    return out, n_roots, len_sel2_vec







def build_features_optimized(cosmos_cat, zbin, node_features, sample_fraction=0.3):
    """
    Optimized version of build_features.
    We find progenitors candidates in next zbin for a given galaxy with node_features
    """
    
    # 1. Filter: Galaxies within zbin
    mask_z = (cosmos_cat['zpdf_med'] > zbin[0]) & (cosmos_cat['zpdf_med'] < zbin[1])
    cosmos_slice = cosmos_cat[mask_z].copy()
    
    if cosmos_slice.empty:
        print(f"In build_features_optimized for zbin {zbin} - Did not find any galaxies in zbin specified.")
        # Return empty
        return {k: [] for k in node_features}, 0, [0]*len(node_features['x'])

    # Extract candidate columns to arrays
    c_mass = cosmos_slice['mass_CIGALE'].values
    c_sfr = cosmos_slice['sfr_CIGALE'].values
    c_rad = cosmos_slice['log_radius_kpc'].values
    c_z = cosmos_slice['zpdf_med'].values
    c_ra = cosmos_slice['ra'].values
    c_dec = cosmos_slice['dec'].values
    c_sersic = cosmos_slice['sersic'].values
    c_bovert = cosmos_slice['bovert'].values
    c_morph = cosmos_slice['morphology'].values
    c_id = cosmos_slice['id'].values

    # Dictionary to store output
    out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id']}
    sel2_len_vec = [] # Aquí guardaremos el tamaño REAL de los hijos generados
    
    # Descendant data (anterior zbin)
    ids_in = node_features['id']
    xs_in = node_features['x']
    ts_in = node_features['t']
    ras_in = node_features['ra']
    decs_in = node_features['dec']
    sersics_in = node_features['sersic']
    boverts_in = node_features['bovert']
    morphs_in = node_features['morphology']
    
    n_chunks = len(xs_in)

    # 2. Main loop
    for i in range(n_chunks):
        
        mass_last = xs_in[i][-1, 0]
        
        # FILTER: Find candidates that fulfill mass condition
        # Condition: mass root - 1.5 < mass candidato < mass_root + 1.5
        mass_mask = np.abs(c_mass - mass_last) < 1.5
        candidate_indices = np.where(mass_mask)[0]
        n_candidates = len(candidate_indices)
        
        actual_prog_count = 0 
        
        # FILTER: Sampling a fraction of those candidates (by default sample_fraction=0.03)
        if n_candidates > 0:
            size_sample = int(n_candidates * sample_fraction)
            
            if size_sample > 0:
                # Choose index
                chosen_indices = np.random.choice(candidate_indices, size=size_sample, replace=False)
                actual_prog_count = len(chosen_indices)
                
                # Prepare descendant data (galaxy in anterior zbin)
                p_x = xs_in[i]
                p_t = ts_in[i]
                p_ra = ras_in[i]
                p_dec = decs_in[i]
                p_ser = sersics_in[i]
                p_bov = boverts_in[i]
                p_mor = morphs_in[i]
                p_id = ids_in[i]

                # Build branches
                for idx in chosen_indices:
                    new_row_x = np.array([c_mass[idx], c_rad[idx], c_sfr[idx]], dtype=np.float32)
                    new_row_t = np.array([1.0 / (1.0 + c_z[idx])], dtype=np.float32)
                    
                    new_row_ra = np.array([c_ra[idx]], dtype=np.float32)
                    new_row_dec = np.array([c_dec[idx]], dtype=np.float32)
                    new_row_ser = np.array([c_sersic[idx]], dtype=np.float32)
                    new_row_bov = np.array([c_bovert[idx]], dtype=np.float32)
                    new_row_mor = np.array([c_morph[idx]], dtype=np.float32)
                    new_row_id = np.array([c_id[idx]], dtype=np.float32)

                    out['x'].append(np.vstack([p_x, new_row_x]))
                    out['t'].append(np.vstack([p_t, new_row_t]))
                    out['ra'].append(np.vstack([p_ra, new_row_ra]))
                    out['dec'].append(np.vstack([p_dec, new_row_dec]))
                    out['sersic'].append(np.vstack([p_ser, new_row_ser]))
                    out['bovert'].append(np.vstack([p_bov, new_row_bov]))
                    out['morphology'].append(np.vstack([p_mor, new_row_mor]))
                    out['id'].append(np.vstack([p_id, new_row_id]))
        
        sel2_len_vec.append(actual_prog_count)

    print(f'So far everything went okay for build_features_optimized for zbin {float(zbin[0]), float(zbin[1])}')
    return out, n_chunks, sel2_len_vec





# Version actualizada de log_likelihood_obs_optimized 02/03:

def log_likelihood_obs_optimized(
        model: torch.nn.Module, 
        batch: Union[Tuple[torch.Tensor], torch.Tensor, dict],
        batch_size: int = 1000,  # Valor recomendado para GPUs de ~12GB
        to_numpy: bool = True,
        device: str = None
    ) -> Union[Tensor, np.ndarray]:
    """
    Calculate log-likelihood
    
    Evaluate a batch of trees generated previously to determine how likely are they according to
    trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Recurrent model 
    batch : Tuple[Tensor]
        Data converted to tensor for the model

    to_numpy : bool, opcional
        If True, converts result from PyTorch tensor to NumPy array. (Default: True)
    device : str, opcional
        Device where to execute ('cpu' or 'cuda'). If None, will try to use GPU is available

    Returns
    -------
    lp : Union[Tensor, np.ndarray]
        Vector with log-likelihood values for each sequence evaluated.
    """
    
    # 1. Choose device (GPU vs CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda' 
            torch.cuda.empty_cache()

        else: device = 'cpu'
        
    model = model.to(device)
    model.eval()


    # Move data to same device as model
    # Determinamos el número total de muestras (galaxias/hijos)
    if isinstance(batch, (tuple, list)):
        num_samples = batch[0].shape[0]
    elif isinstance(batch, dict):
        num_samples = next(iter(batch.values())).shape[0]
    else:
        num_samples = batch.shape[0]

    all_lps = []

    # 2. Evaluación por trozos (Mini-batches)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Extraer sub-batch y enviarlo a la GPU solo en este momento
            if isinstance(batch, (tuple, list)):
                sub_batch = tuple(b[i : i + batch_size].to(device) for b in batch)
            elif isinstance(batch, dict):
                sub_batch = {k: v[i : i + batch_size].to(device) for k, v in batch.items()}
            else:
                sub_batch = batch[i : i + batch_size].to(device)

            # Calcular probabilidad
            lp_sub = model.log_prob(sub_batch, return_context=False)
            
            # Mover resultado a CPU inmediatamente para liberar RAM de la GPU
            all_lps.append(lp_sub.cpu())
            
            # Limpieza explícita de referencias temporales
            del sub_batch
            if device == 'cuda' and i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()

    # 3. Concatenar resultados
    lp = torch.cat(all_lps)

    # 3. Convert to NumPy if solicited
    if to_numpy:
        # We use .cpu() first in case tensor was in gpu
        lp = lp.detach().numpy()

    return lp






def get_maxlike_descendant_final(l_numpy, node_features, num_chunks, chunk_size):
    """
    Optimized and final version of get_maxlike_descendant
    Choose progenitor with max likelihood 
    """
    
    # 1. Safety Check - Making sure l_numpy has same size as chunk_size
    total_hijos = np.sum(chunk_size)
    total_probs = len(l_numpy)
    

    # 1. DETERMINAR LA LONGITUD DE LA SECUENCIA
    # Si tenemos 25372 probs para 12686 hijos, seq_len es 2.
    if total_hijos > 0:
        seq_len = total_probs // total_hijos
        
        # 2. EXTRAER SOLO LA PROBABILIDAD FINAL DE CADA SECUENCIA
        # l_numpy viene como [P_hijo1_paso1, P_hijo1_paso2, P_hijo2_paso1, P_hijo2_paso2...]
        # Queremos los índices: seq_len - 1, (2*seq_len) - 1, etc.
        indices_finales = np.arange(total_hijos) * seq_len + (seq_len - 1)
        l_numpy_filtrado = l_numpy[indices_finales]
    else:
        l_numpy_filtrado = l_numpy

    # 3. VECTORIZACIÓN PARA ENCONTRAR EL MEJOR HIJO
    parent_ids = np.repeat(np.arange(num_chunks), chunk_size)
    
    # Create temporal DataFrame
    df = pd.DataFrame({
        'likelihood': l_numpy_filtrado,
        'parent_id': parent_ids,
        'global_index': np.arange(len(l_numpy_filtrado))
    })
    
    # Find the global index of max likelihood for each progenitor
    # idxmax gives us index for DF where max is
    best_indices = df.loc[df.groupby('parent_id')['likelihood'].idxmax(), 'global_index'].values
    best_indices.sort()

    # 3. Data extraction
    keys_to_extract = ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id']
    node_features_updated = {}

    for key in keys_to_extract:
        original_list = node_features[key]
        # Selecting winning indices
        selected_items = [original_list[i] for i in best_indices]
        node_features_updated[key] = selected_items


    print(f'So far everything went okay for get_maxlike_descendant_final for this zbin')
    return node_features_updated





# --- PART 2 ---- 





# Load the trained model from a checkpoint file
checkpoint_path = "/scratch/lmarrero-ext/CEERS_train/proj/TNGEagleSimba_mass_size_gt9/SFR_val/last-v1.ckpt"  # Specify the path to your checkpoint file
loaded_model = DataModule.load_from_checkpoint(checkpoint_path,map_location='cpu', weights_only=False)
# Set the model to evaluation mode (important if you have dropout or batch normalization layers)
loaded_model.eval()


cosmos_data_path = "/scratch/lmarrero-ext/likelihood_COSMOS_SFR/"
cosmos_cat = pd.read_csv(cosmos_data_path+"COSMOSWeb_Laura_processed.csv") # Data from COSMOS-WEB, converted from .fits to .csv in florah_eval_SFR.ipynb

nfm_data_path = "/scratch/lmarrero-ext/likelihood_COSMOS_SFR/node_features_morphology/"


redshifts = np.array([1.,1.5,2,2.5,3.5,4.5,6])
mass_bin = [[9.8,10],[10,10.2],[10.2,10.4],[10.4,10.6],[10.6,10.8],[10.8,11],[11,12]]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Iniciando ejecución en dispositivo: {device} ---")

for m in mass_bin:
    print(f"\n{'='*60}")
    print(f"Iniciando mass_bin: {m}")
    print(f"{'='*60}")

    # First we build root and find best candidate for progenitor in next redshift bin
    node_features, n_chunks, chunk_size = build_roots_optimized(cosmos_cat, m, nsamples=100) # Select root in zbin = (0, 0.5) + candidate for progenitor in zbin = (0.5, 1)
    loaded_model.to('cpu')
    preprocessed_node_features = loaded_model.transform(node_features, fit=False) 
    l  = log_likelihood_obs_optimized(loaded_model, preprocessed_node_features, device=device) # Calculate likelihood for every pair
    node_features = get_maxlike_descendant_final(l, node_features, n_chunks, chunk_size) # Chooses the galaxy with higher likelihood


    # Loop to find progenitors in the following bins
    for zmin,zmax in zip(redshifts[:-1],redshifts[1:]):
        print(f"Iteración z_bin : {zmin} -> {zmax}")

        node_features, n_chunks, chunk_size = build_features_optimized(cosmos_cat,[zmin,zmax],node_features)
        loaded_model.to('cpu')
        preprocessed_node_features = loaded_model.transform(node_features, fit=False)
        l  = log_likelihood_obs_optimized(loaded_model, preprocessed_node_features, device=device)
        node_features = get_maxlike_descendant_final(l,node_features, n_chunks, chunk_size)

# Store node_features
    with open(nfm_data_path+'node_features_morphology'+str(m[0])+'_'+str(m[1])+'.pkl', 'wb') as outfile:
        pickle.dump(node_features, outfile)
    