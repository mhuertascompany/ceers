import sys
print("--- PYTHON HAS OFFICIALLY STARTED THE SCRIPT ---", flush=True)


import numpy as np
print('se cargo numpy')
import pandas as pd
print('scipy cargando...')
from scipy.interpolate import interp1d
print('se ha cargao scipy')


import h5py  
import pickle

import pdb
 
import torch
from typing import Optional, Tuple, Union
from torch import Tensor

from operator import itemgetter

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value, LambdaCDM
cosmo = Planck13

import os

sys.path.append('/scratch/lmarrero-ext/florah/src')

from florah.models.rnn_model.rnn_generator import DataModule 
print('Libraries loaded')

# --- Functions ---
def calcular_bordes_bins_2d(centros):
    # 1. Calculamos los puntos medios entre las columnas adyacentes
    puntos_medios = (centros[:, 1:] + centros[:, :-1]) / 2.0
    
    # 2. Extrapolamos el primer borde de cada fila
    # (Restamos la mitad de la distancia entre el primer y el segundo centro)
    primer_borde = centros[:, 0:1] - (centros[:, 1:2] - centros[:, 0:1]) / 2.0
    
    # 3. Extrapolamos el último borde de cada fila
    # (Sumamos la mitad de la distancia entre el último y el penúltimo centro)
    ultimo_borde = centros[:, -1:] + (centros[:, -1:] - centros[:, -2:-1]) / 2.0
    
    # 4. Concatenamos todo a lo largo del eje 1 (columnas)
    bordes = np.hstack((primer_borde, puntos_medios, ultimo_borde))
    
    return bordes


def parse_string_array(s):
        if isinstance(s, str):
            # Remove brackets and split by whitespace
            return [float(x) for x in s.replace('[', '').replace(']', '').split()]
        return s  # Returns as-is if it's already a valid list/array


def build_roots_optimized(cosmos_cat, mass_bin, nsamples=100, zbins=[0, 0.5], sample_fraction=1):
    """
    Optimized version of build_roots.
    Generates initial pairs of galaxies (root -> progenitor candidate), to initialize trees
    """
    
    # ---------------------------------------------------------
    # 1. Prepare root pool
    # ---------------------------------------------------------
    mask_roots = (
        (cosmos_cat['zpdf_med'] > zbins[0]) & 
        (cosmos_cat['zpdf_med'] < zbins[1]) & 
        (cosmos_cat['mass_CIGALE'] > mass_bin[0]) & 
        (cosmos_cat['mass_CIGALE'] < mass_bin[1])
    )
    pool_roots = cosmos_cat[mask_roots].copy()

    # ---------------------------------------------------------
    # 2. Root selection (w nsamples)
    # ---------------------------------------------------------
    if len(pool_roots) < nsamples:
        pool_roots = pool_roots.sample(frac=1, random_state=42)
    else:
        pool_roots = pool_roots.sample(n=nsamples, random_state=42)
        
    n_roots = len(pool_roots)
    print('n_roots', n_roots)

    # If unable to find any root galaxy, return empty safely with all required keys
    if pool_roots.empty:
        print(f"En build_roots_optimized con mass_bin {mass_bin} - Advertencia: No se encontraron galaxias raíces.")
        empty_out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id', 'track_idx']}
        return empty_out, 0, [], []

    # --- Defining ZBINS ---
    parsed_time_list = pool_roots['time'].apply(parse_string_array).tolist()
    time_matrix = np.array(parsed_time_list, dtype=float)
    cigale_lbt = time_matrix * 1e-3  

    age_root = cosmo.age(pool_roots['zpdf_med'].values).value  
    cigale_time = age_root[:, None] - cigale_lbt 
    max_age = cosmo.age(0).value
    cigale_time = np.clip(cigale_time, 1e-6, max_age - 1e-6)

    # Fast Interpolation Grid for Age -> Redshift conversion
    z_grid = np.linspace(0, 15, 5000) 
    age_grid = cosmo.age(z_grid).value
    age_to_z_interp = interp1d(age_grid[::-1], z_grid[::-1], kind='cubic', fill_value="extrapolate")
    cigale_z = age_to_z_interp(cigale_time)

    redshifts = calcular_bordes_bins_2d(cigale_z)

    # ---------------------------------------------------------
    # 3. Prepare candidate pool (Broad global filter)
    # ---------------------------------------------------------
    min_req_z = np.min(redshifts[:, 1])
    max_req_z = np.max(redshifts[:, 2])
    
    mask_candidates_global = (
        (cosmos_cat['zpdf_med'] >= min_req_z) & 
        (cosmos_cat['zpdf_med'] <= max_req_z)
    )
    pool_candidates = cosmos_cat[mask_candidates_global].copy()
    
    if pool_candidates.empty:
        print(f"Advertencia: No se encontraron candidatos globales para los rangos de Z.")
        empty_out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id', 'track_idx']}
        return empty_out, n_roots, [], redshifts

    # --- EXTRACCIÓN A NUMPY ---
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

    # FIX: Initialize the output structure with 'track_idx' included
    out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id', 'track_idx']}
    len_sel2_vec = []
    total_pairs = 0
    rng = np.random.default_rng(seed=42)

    # ---------------------------------------------------------
    # 4. Main loop (iterate over each root)
    # ---------------------------------------------------------
    for i in range(n_roots):
        
        mass_root = r_mass[i]
        mass_root10 = 10**mass_root
        u1 = mass_root10/2
        u2 = mass_root10/1.2
        x1 = np.log10((mass_root10 + u1)/mass_root10)
        x2 = np.log10(mass_root10/(mass_root10 - u2))
        
        z_min_root = redshifts[i, 1]
        z_max_root = redshifts[i, 2]
        
        # Combined filtering per branch matrix row
        mask_matches = (
            ((mass_root + x1) > c_mass) & (c_mass > (mass_root - x2)) &
            (c_z > z_min_root) & (c_z < z_max_root)
        )

        match_indices = np.where(mask_matches)[0]
        n_matches = len(match_indices)
        
        if n_matches > 0:
            n_sample = int(n_matches * sample_fraction)
            
            if n_sample > 0:
                chosen_indices = rng.choice(match_indices, size=n_sample, replace=False)
                
                # Pre-calculate Root features
                row0_x = [r_mass[i], r_rad[i], r_sfr[i]]
                row0_t = [1.0 / (1.0 + r_z[i])]
                row0_ra = [r_ra[i]]
                row0_dec = [r_dec[i]]
                row0_ser = [r_ser[i]]
                row0_bov = [r_bov[i]]
                row0_mor = [r_mor[i]]
                row0_id = [r_id[i]]
                row0_track = [i] # Tracking index equals the root sequence row

                for idx in chosen_indices:
                    row1_x = [c_mass[idx], c_rad[idx], c_sfr[idx]]
                    row1_t = [1.0 / (1.0 + c_z[idx])]
                    row1_ra = [c_ra[idx]]
                    row1_dec = [c_dec[idx]]
                    row1_ser = [c_sersic[idx]]
                    row1_bov = [c_bovert[idx]]
                    row1_mor = [c_morph[idx]]
                    row1_id = [c_id[idx]]
                    row1_track = [i]
                    
                    out['x'].append(np.array([row0_x, row1_x], dtype=np.float32))
                    out['t'].append(np.array([row0_t, row1_t], dtype=np.float32)) 
                    out['ra'].append(np.array([row0_ra, row1_ra], dtype=np.float32))
                    out['dec'].append(np.array([row0_dec, row1_dec], dtype=np.float32))
                    out['sersic'].append(np.array([row0_ser, row1_ser], dtype=np.float32))
                    out['bovert'].append(np.array([row0_bov, row1_bov], dtype=np.float32))
                    out['morphology'].append(np.array([row0_mor, row1_mor], dtype=np.float32))
                    out['id'].append(np.array([row0_id, row1_id], dtype=np.float32))
                    # FIX: Safely append the tracking history block with shape (2, 1)
                    out['track_idx'].append(np.array([row0_track, row1_track], dtype=np.int32))
                    
                    total_pairs += 1
            
            len_sel2_vec.append(n_sample) 
        else:
            len_sel2_vec.append(0)

    print(f'Done! Successfully processed mass_bin {mass_bin} with {total_pairs} total pairs.')
    return out, n_roots, len_sel2_vec, redshifts



def build_features_optimized(cosmos_cat, zmin_vector, zmax_vector, node_features, sample_fraction=1):
    """
    Optimized version of build_features.
    Finds progenitor candidates using customized redshift vectors for each individual tracking branch.
    """
    # 1. Broad Global Filter: Narrows down cosmos_cat to speed up the loop
    global_min_z = min(np.min(zmin_vector), 14)
    global_max_z = min(np.max(zmax_vector), 15)
    
    mask_broad_z = (cosmos_cat['zpdf_med'] >= global_min_z) & (cosmos_cat['zpdf_med'] <= global_max_z)
    cosmos_slice = cosmos_cat[mask_broad_z].copy()
    
    if cosmos_slice.empty:
        print(f"Warning: Did not find any galaxies globally between z={global_min_z:.2f} and z={global_max_z:.2f}")
        return node_features, n_chunks, sel2_len_vec

    # Fast NumPy Extractions
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

    # Output structure (including tracking index propagation)
    out = {k: [] for k in ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id', 'track_idx']}
    sel2_len_vec = [] 
    
    xs_in = node_features['x']
    ts_in = node_features['t']
    ras_in = node_features['ra']
    decs_in = node_features['dec']
    sersics_in = node_features['sersic']
    boverts_in = node_features['bovert']
    morphs_in = node_features['morphology']
    ids_in = node_features['id']
    tracks_in = node_features['track_idx'] # Extracted track row mapping
    
    n_chunks = len(xs_in)
    
    # Initialize the Random Number Generator OUTSIDE the loop for efficiency
    rng = np.random.default_rng(seed=42)

    # 2. Main loop over existing active branches
    for i in range(n_chunks):
        
        # Pull the absolute tracking row index to lookup the correct redshift limits
        # tracking array has shape (History_Length, 1), we look at the root value at index 0
        track_row_idx = int(tracks_in[i][0, 0]) 
        
        zmin_this_galaxy = zmin_vector[track_row_idx]
        zmax_this_galaxy = zmax_vector[track_row_idx]
        
        # Mass calculation boundaries
        mass_last = xs_in[i][-1, 0]
        mass_last10 = 10**mass_last
        u1 = mass_last10 / 0.1
        u2 = mass_last10 / 1.1
        x1 = np.log10((mass_last10 + u1) / mass_last10)
        x2 = np.log10(mass_last10 / (mass_last10 - u2))
        
        # COMBINED FILTER: Match mass conditions AND specific redshift window for this tracking line
        mass_mask = (mass_last + x1 > c_mass) & (c_mass > mass_last - x2)
        z_mask = (c_z > zmin_this_galaxy) & (c_z < zmax_this_galaxy)
        
        candidate_indices = np.where(mass_mask & z_mask)[0]
        n_candidates = len(candidate_indices)
        
        
        actual_prog_count = 0 
        
        if n_candidates > 0:
            size_sample = int(n_candidates * sample_fraction)
            
            if size_sample > 0:
                # FIX: Use 'size_sample' correctly here
                chosen_indices = rng.choice(candidate_indices, size=size_sample, replace=False)
                actual_prog_count = len(chosen_indices)
                
                # Prepare current branch history data
                p_x = xs_in[i]
                p_t = ts_in[i]
                p_ra = ras_in[i]
                p_dec = decs_in[i]
                p_ser = sersics_in[i]
                p_bov = boverts_in[i]
                p_mor = morphs_in[i]
                p_id = ids_in[i]
                p_track = tracks_in[i]

                # Append new progenitor steps onto the branch matrices
                for idx in chosen_indices:
                    new_row_x = np.array([c_mass[idx], c_rad[idx], c_sfr[idx]], dtype=np.float32)
                    new_row_t = np.array([1.0 / (1.0 + c_z[idx])], dtype=np.float32)
                    new_row_ra = np.array([c_ra[idx]], dtype=np.float32)
                    new_row_dec = np.array([c_dec[idx]], dtype=np.float32)
                    new_row_ser = np.array([c_sersic[idx]], dtype=np.float32)
                    new_row_bov = np.array([c_bovert[idx]], dtype=np.float32)
                    new_row_mor = np.array([c_morph[idx]], dtype=np.float32)
                    new_row_id = np.array([c_id[idx]], dtype=np.float32)
                    new_row_track = np.array([track_row_idx], dtype=np.int32)

                    out['x'].append(np.vstack([p_x, new_row_x]))
                    out['t'].append(np.vstack([p_t, new_row_t]))
                    out['ra'].append(np.vstack([p_ra, new_row_ra]))
                    out['dec'].append(np.vstack([p_dec, new_row_dec]))
                    out['sersic'].append(np.vstack([p_ser, new_row_ser]))
                    out['bovert'].append(np.vstack([p_bov, new_row_bov]))
                    out['morphology'].append(np.vstack([p_mor, new_row_mor]))
                    out['id'].append(np.vstack([p_id, new_row_id]))
                    out['track_idx'].append(np.vstack([p_track, new_row_track]))
        else: 
            # Prepare current branch history data
                p_x = xs_in[i]
                p_t = ts_in[i]
                p_ra = ras_in[i]
                p_dec = decs_in[i]
                p_ser = sersics_in[i]
                p_bov = boverts_in[i]
                p_mor = morphs_in[i]
                p_id = ids_in[i]
                p_track = tracks_in[i]

                out['x'].append(p_x)
                out['t'].append(p_t)
                out['ra'].append(p_ra)
                out['dec'].append(p_dec)
                out['sersic'].append(p_ser)
                out['bovert'].append(p_bov)
                out['morphology'].append(p_mor)
                out['id'].append(p_id)
                out['track_idx'].append(p_track)


        sel2_len_vec.append(actual_prog_count)

    print(f'Iteration complete. build_features_optimized processed {n_chunks} branches.')
    return out, n_chunks, sel2_len_vec



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
    best_indices = np.sort(best_indices)

    # 3. Data extraction
    keys_to_extract = ['x', 't', 'ra', 'dec', 'sersic', 'bovert', 'morphology', 'id', 'track_idx']
    node_features_updated = {}

    for key in keys_to_extract:
        original_list = node_features[key]
        # Selecting winning indices
        selected_items = [original_list[i] for i in best_indices]
        node_features_updated[key] = selected_items


    print(f'So far everything went okay for get_maxlike_descendant_final for this zbin')
    return node_features_updated


# --- PART 2 ---- 

print('Loading model')
# Load the trained model from a checkpoint file
checkpoint_path = "/scratch/lmarrero-ext/CEERS_train/proj/TNGEagleSimba_mass_size_gt9/SFR_val/last-v1.ckpt"  # Specify the path to your checkpoint file
loaded_model = DataModule.load_from_checkpoint(checkpoint_path,map_location='cpu', weights_only=False)
# Set the model to evaluation mode (important if you have dropout or batch normalization layers)
loaded_model.eval()


print("Loading cosmos catalog safely...")
cosmos_data_path = "/scratch/lmarrero-ext/likelihood_COSMOS_SFR/"
cosmos_cat = pd.read_csv(cosmos_data_path+"COSMOSWeb_Laura_processed_SFH.csv", engine='python') # Data from COSMOS-WEB, converted from .fits to .csv in florah_eval_SFR.ipynb
print("Catalog loaded successfully!")

nfm_data_path = "/scratch/lmarrero-ext/likelihood_COSMOS_SFR/node_features_morphology/"


mass_bin = [[9.8,10],[10,10.2],[10.2,10.4],[10.4,10.6],[10.6,10.8],[10.8,11],[11,12]]
sample_fraction = np.array([1, 1, 1, 1, 1, 1, 1, 0.5, 0.5])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Iniciando ejecución en dispositivo: {device} ---")

for m in mass_bin:
    print(f"\n{'='*60}")
    print(f"Iniciando mass_bin: {m}")
    print(f"{'='*60}")

    # First we build root and find best candidate for progenitor in next redshift bin
    node_features, n_chunks, chunk_size, redshifts = build_roots_optimized(cosmos_cat, m, nsamples=10) # Select root in zbin = (0, 0.5) + candidate for progenitor in zbin = (0.5, 1)
    loaded_model.to('cpu')
    preprocessed_node_features = loaded_model.transform(node_features, fit=False) 
    l  = log_likelihood_obs_optimized(loaded_model, preprocessed_node_features, device=device) # Calculate likelihood for every pair
    node_features = get_maxlike_descendant_final(l, node_features, n_chunks, chunk_size) # Chooses the galaxy with higher likelihood

    # Loop to find progenitors in the following bins
    # redshifts shape is (n_roots, n_bins)
    num_bins = redshifts.shape[1]

    for bin_idx in range(2, num_bins - 1):
        # Extract the custom redshift boundaries for ALL tracks at this specific step
        # These are now arrays/vectors of shape (n_roots,) instead of single numbers!
        zmin_vector = redshifts[:, bin_idx]
        zmax_vector = redshifts[:, bin_idx + 1]
        
        print(f"Iteración z_bin columna: {bin_idx} -> {bin_idx + 1}")
        print(f"Rango de redshifts en este paso: {zmin_vector.min():.2f} a {zmax_vector.max():.2f}")

        print('---Tamaño de node_features antes de build_features:', len(node_features['bovert']))
        node_features, n_chunks, chunk_size = build_features_optimized(cosmos_cat, zmin_vector, zmax_vector, node_features, sample_fraction=sample_fraction[bin_idx])
        print('---Tamaño de node_features dps:' , len(node_features['bovert']))
        if len(node_features['bovert']) == 0:
            print('Rompí el bucle')
            break
        print('nchunks', n_chunks)
        if n_chunks == len(node_features['bovert']):
            print('igualdad')
            break
        loaded_model.to('cpu')
        preprocessed_node_features = loaded_model.transform(node_features, fit=False)
        l = log_likelihood_obs_optimized(loaded_model, preprocessed_node_features, device=device)
        node_features = get_maxlike_descendant_final(l, node_features, n_chunks, chunk_size)
        print('---Tamaño de node_features dps del bucle de zbin:' , len(node_features['bovert']))

    node_features['formation_history_zbins'] = redshifts
# Store node_features
    with open(nfm_data_path+'node_features_morphology'+str(m[0])+'_'+str(m[1])+'.pkl', 'wb') as outfile:
        pickle.dump(node_features, outfile)
