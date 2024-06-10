import numpy as np
import pandas as pd
from scipy.stats import betabinom
import os
import time
from multiprocessing import Pool, cpu_count

def sample_posterior(N, pmf, n_samples=1):
    return np.random.choice(np.arange(len(pmf)), size=n_samples, p=pmf/np.sum(pmf))

def process_chunk(args):
    chunk, filter, alpha_feature_pred, alpha_smooth_pred, alpha_artifact_pred, alpha_edgeon_pred, alpha_else_pred, alpha_strong_pred, alpha_weak_pred, alpha_none_pred, alpha_spiral_pred,alpha_spiral_else_pred,alpha_merger_major_pred,alpha_merger_merging_pred,alpha_merger_minor_pred,alpha_merger_none_pred,alpha_clump_pred,alpha_clump_no_pred,num_runs, num_vols, num_models, num_classes = args

    results = []

    for i in chunk.index:
        feature_count = np.zeros(num_runs, dtype=int)
        edgeon_count = np.zeros(num_runs, dtype=int)
        bar_count = np.zeros(num_runs, dtype=int)
        spiral_count = np.zeros(num_runs, dtype=int)
        clump_count = np.zeros(num_runs, dtype=int)
        merger_count = np.zeros(num_runs, dtype=int)

        pmf_feature = np.zeros((num_models * num_classes, num_vols + 1))
        for j in range(num_models):
            a_feature = alpha_feature_pred[j][i]
            b_feature = alpha_smooth_pred[j][i] + alpha_artifact_pred[j][i]
            if isinstance(a_feature, np.ndarray) and isinstance(b_feature, np.ndarray):
                for l in range(num_classes):
                    pmf_feature[j * num_classes + l, :] = betabinom.pmf(range(num_vols + 1), num_vols, a_feature[l], b_feature[l])
            else:
                pmf_feature[j, :] = betabinom.pmf(range(num_vols + 1), num_vols, a_feature, b_feature)
        mean_pmf_feature = np.mean(pmf_feature, axis=0)
        feature_count = sample_posterior(num_vols, mean_pmf_feature, n_samples=num_runs)

        pmf_smooth = np.zeros((num_models * num_classes, num_vols + 1))
        for j in range(num_models):
            a_smooth = alpha_smooth_pred[j][i]
            b_smooth = alpha_feature_pred[j][i] + alpha_artifact_pred[j][i]
            if isinstance(a_smooth, np.ndarray) and isinstance(b_smooth, np.ndarray):
                for l in range(num_classes):
                    pmf_smooth[j * num_classes + l, :] = betabinom.pmf(range(num_vols + 1), num_vols, a_smooth[l], b_smooth[l])
            else:
                pmf_smooth[j, :] = betabinom.pmf(range(num_vols + 1), num_vols, a_smooth, b_smooth)
        mean_pmf_smooth = np.mean(pmf_smooth, axis=0)
        smooth_count = sample_posterior(num_vols, mean_pmf_smooth, n_samples=num_runs)

        

        for k in range(num_runs):
            N_FEATURE = feature_count[k]
            N_SMOOTH=smooth_count[k]
            pmf_edgeon = np.zeros((num_models * num_classes, N_FEATURE + 1))
            for j in range(num_models):
                a_disk = alpha_edgeon_pred[j][i]
                b_disk = alpha_else_pred[j][i]
                if isinstance(a_disk, np.ndarray) and isinstance(b_disk, np.ndarray):
                    for l in range(num_classes):
                        pmf_edgeon[j * num_classes + l, :] = betabinom.pmf(range(N_FEATURE + 1), N_FEATURE, a_disk[l], b_disk[l])
                else:
                    pmf_edgeon[j, :] = betabinom.pmf(range(N_FEATURE + 1), N_FEATURE, a_disk, b_disk)
            mean_pmf_edgeon = np.mean(pmf_edgeon, axis=0)
            edgeon_count[k] = sample_posterior(N_FEATURE, mean_pmf_edgeon)

            N_FACEON = feature_count[k] - edgeon_count[k]
            N_EDGEON = edgeon_count[k]
            pmf_bar = np.zeros((num_models * num_classes, N_FACEON + 1))
            for j in range(num_models):
                a_bar = alpha_strong_pred[j][i] + alpha_weak_pred[j][i]
                b_bar = alpha_none_pred[j][i]
                if isinstance(a_bar, np.ndarray) and isinstance(b_bar, np.ndarray):
                    for l in range(num_classes):
                        pmf_bar[j * num_classes + l, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_bar[l], b_bar[l])
                else:
                    pmf_bar[j, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_bar, b_bar)
            mean_pmf_bar = np.mean(pmf_bar, axis=0)
            bar_count[k] = sample_posterior(N_FACEON, mean_pmf_bar)

            N_BAR = bar_count[k]
            pmf_spiral = np.zeros((num_models * num_classes, N_FACEON + 1))
            for j in range(num_models):
                a_spiral = alpha_spiral_pred[j][i] 
                b_spiral = alpha_spiral_else_pred[j][i]
                if isinstance(a_spiral, np.ndarray) and isinstance(b_spiral, np.ndarray):
                    for l in range(num_classes):
                        pmf_spiral[j * num_classes + l, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_spiral[l], b_spiral[l])
                else:
                    pmf_spiral[j, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_spiral, b_spiral)
            mean_pmf_spiral = np.mean(pmf_spiral, axis=0)
            spiral_count[k] = sample_posterior(N_FACEON, mean_pmf_spiral)

            N_SPIRAL=spiral_count[k]
            N_NO_SPIRAL = N_BAR-N_SPIRAL
            pmf_clumps = np.zeros((num_models * num_classes, N_NO_SPIRAL+N_EDGEON + 1))
            for j in range(num_models):
                a_clump = alpha_clump_pred[j][i] 
                b_clump = alpha_clump_no_pred[j][i]
                if isinstance(a_clump, np.ndarray) and isinstance(b_clump, np.ndarray):
                    for l in range(num_classes):
                        pmf_clumps[j * num_classes + l, :] = betabinom.pmf(range(N_NO_SPIRAL+N_EDGEON + 1), N_NO_SPIRAL+N_EDGEON, a_clump[l], b_clump[l])
                else:
                    pmf_clumps[j, :] = betabinom.pmf(range(N_NO_SPIRAL+N_EDGEON + 1), N_NO_SPIRAL+N_EDGEON, a_clump, b_clump)
            mean_pmf_clump = np.mean(pmf_clumps, axis=0)
            clump_count[k] = sample_posterior(N_NO_SPIRAL+N_EDGEON , mean_pmf_clump)

            pmf_mergers = np.zeros((num_models * num_classes, N_SMOOTH + N_NO_SPIRAL + N_EDGEON + 1))
            for j in range(num_models):
                a_merger = alpha_merger_major_pred[j][i]+alpha_merger_merging_pred[j][i]+alpha_merger_minor_pred[j][i]
                b_merger = alpha_merger_none_pred[j][i]
                if isinstance(a_merger, np.ndarray) and isinstance(b_merger, np.ndarray):
                    for l in range(num_classes):
                        pmf_mergers[j * num_classes + l, :] = betabinom.pmf(range(N_NO_SPIRAL+N_EDGEON + 1), N_NO_SPIRAL+N_EDGEON, a_merger[l], b_merger[l])
                else:
                    pmf_mergers[j, :] = betabinom.pmf(range(N_NO_SPIRAL+N_EDGEON + 1), N_NO_SPIRAL+N_EDGEON, a_merger, b_merger)
            mean_pmf_merger = np.mean(pmf_mergers, axis=0)
            merger_count[k] = sample_posterior(N_NO_SPIRAL+N_EDGEON , mean_pmf_merger)


        results.append({
            'id': chunk.loc[i, 'id_str'],
            'feature_count': np.array2string(feature_count, separator=', '),
            'smooth_count': np.array2string(smooth_count, separator=', '),
            'edgeon_count': np.array2string(edgeon_count, separator=', '),
            'bar_count': np.array2string(bar_count, separator=', '),
            'merger_count': np.array2string(merger_count, separator=', '),
            'spiral_count': np.array2string(spiral_count, separator=', '),
            'clump_count': np.array2string(clump_count, separator=', '),

        })

    return results

def parallel_process(pred, filter, alpha_feature_pred, alpha_smooth_pred, alpha_artifact_pred, alpha_edgeon_pred, alpha_else_pred, alpha_strong_pred, alpha_weak_pred, alpha_none_pred, alpha_spiral_pred,alpha_spiral_else_pred,alpha_merger_major_pred,alpha_merger_merging_pred,alpha_merger_minor_pred,alpha_merger_none_pred,alpha_clump_pred,alpha_clump_no_pred,num_runs, num_vols, num_models, num_classes):
    num_cpus = 20  # Set this to match the --cpus-per-task in your SLURM script
    chunk_size = len(pred) // num_cpus
    chunks = [pred.iloc[i:i + chunk_size] for i in range(0, len(pred), chunk_size)]

    args = [(chunk, filter, alpha_feature_pred, alpha_smooth_pred, alpha_artifact_pred, alpha_edgeon_pred, alpha_else_pred, alpha_strong_pred, alpha_weak_pred, alpha_none_pred, alpha_spiral_pred,alpha_spiral_else_pred,alpha_merger_major_pred,alpha_merger_merging_pred,alpha_merger_minor_pred,alpha_merger_none_pred,alpha_clump_pred,alpha_clump_no_pred,num_runs, num_vols, num_models, num_classes) for chunk in chunks]

    with Pool(num_cpus) as pool:
        results = pool.map(process_chunk, args)

    # Flatten list of lists
    results = [item for sublist in results for item in sublist]
    return results

FILTERS = ['f150w', 'f277w', 'f444w']
N_RUNS = 100
N_VOLS = 100
NUM_MODELS = 1  # Number of model predictions
NUM_CLASSES = 5  # Number of classes per model

# Load data
cat_dir = "/n03data/huertas/COSMOS-Web/cats"

for FILTER in FILTERS:
    # Initialize lists for predictions
    alpha_feature_pred = []
    alpha_smooth_pred = []
    alpha_artifact_pred = []

    alpha_edgeon_pred = []
    alpha_else_pred = []

    alpha_strong_pred = []
    alpha_weak_pred = []
    alpha_none_pred = []

    alpha_spiral_pred=[]
    alpha_spiral_else_pred=[]

    alpha_merger_major_pred=[]
    alpha_merger_merging_pred=[]
    alpha_merger_minor_pred=[]
    alpha_merger_none_pred=[]

    alpha_clump_pred=[]
    alpha_clump_no_pred=[]

    # Load predictions for each filter
    for i in range(NUM_MODELS):
        pred_path = os.path.join(cat_dir, f'bars_COSMOS_{FILTER}_m27_effnet.csv')
        pred = pd.read_csv(pred_path)

        alpha_feature_pred.append(pred['t0_smooth_or_featured__features_or_disk_pred'].values)
        alpha_smooth_pred.append(pred['t0_smooth_or_featured__smooth_pred'].values)
        alpha_artifact_pred.append(pred['t0_smooth_or_featured__star_artifact_or_bad_zoom_pred'].values)

        alpha_edgeon_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__yes_edge_on_disk_pred'].values)
        alpha_else_pred.append(pred['t2_could_this_be_a_disk_viewed_edgeon__no_something_else_pred'].values)

        alpha_strong_pred.append(pred['t4_is_there_a_bar__strong_bar_pred'].values)
        alpha_weak_pred.append(pred['t4_is_there_a_bar__weak_bar_pred'].values)
        alpha_none_pred.append(pred['t4_is_there_a_bar__no_bar_pred'].values)

        alpha_strong_pred.append(pred['t4_is_there_a_bar__strong_bar_pred'].values)
        alpha_weak_pred.append(pred['t4_is_there_a_bar__weak_bar_pred'].values)
        alpha_none_pred.append(pred['t4_is_there_a_bar__no_bar_pred'].values)

        alpha_spiral_pred.append(pred['t5_is_there_any_spiral_arm_pattern__yes_pred'].values)
        alpha_spiral_else_pred.append(pred['t5_is_there_any_spiral_arm_pattern__no_pred'].values)

        alpha_merger_major_pred.append(pred['t11_is_the_galaxy_merging_or_disturbed__major_disturbance_pred'].values)
        alpha_merger_merging_pred.append(pred['t11_is_the_galaxy_merging_or_disturbed__merging_pred'].values)
        alpha_merger_minor_pred.append(pred['t11_is_the_galaxy_merging_or_disturbed__minor_disturbance_pred'].values)
        alpha_merger_none_pred.append(pred['t11_is_the_galaxy_merging_or_disturbed__none_pred'].values)

        alpha_clump_pred.append(pred['t12_are_there_any_obvious_bright_clumps__yes_pred'].values)
        alpha_clump_no_pred.append(pred['t12_are_there_any_obvious_bright_clumps__no_pred'].values)


    # Parallel processing
    start = time.time()
    results = parallel_process(pred, FILTER, alpha_feature_pred, alpha_smooth_pred, alpha_artifact_pred, alpha_edgeon_pred, alpha_else_pred, alpha_strong_pred, alpha_weak_pred, alpha_none_pred, alpha_spiral_pred,alpha_spiral_else_pred,alpha_merger_major_pred,alpha_merger_merging_pred,alpha_merger_minor_pred,alpha_merger_none_pred,alpha_clump_pred,alpha_clump_no_pred,N_RUNS, N_VOLS, NUM_MODELS, NUM_CLASSES)
    
    # Convert results to DataFrame
    sampling_result = pd.DataFrame(results)

    # Save results to CSV
    output_path = os.path.join(cat_dir, f'gzoo_COSMOS_{FILTER}_effnet_m27_sampling.csv')
    sampling_result.to_csv(output_path, index=False)

    end = time.time()
    print(f'Sampling completed for filter: {FILTER}, results saved to {output_path}, Time elapsed: {end - start}')
