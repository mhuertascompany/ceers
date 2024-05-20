import numpy as np
import pandas as pd
from scipy.stats import betabinom
import os
import time

def sample_posterior(N, pmf, n_samples=1):
    return np.random.choice(np.arange(len(pmf)), size=n_samples, p=pmf/np.sum(pmf))

FILTERS = ['f150w', 'f277w', 'f444w']
N_RUNS = 100
N_VOLS = 100
NUM_MODELS = 3  # Number of model predictions
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

    # Precompute and store results in lists
    results = []

    start = time.time()

    #for i in range(len(pred)):
    for i in range(100):
        feature_count = np.zeros(N_RUNS, dtype=int)
        edgeon_count = np.zeros(N_RUNS, dtype=int)
        bar_count = np.zeros(N_RUNS, dtype=int)

        pmf_feature = np.zeros((NUM_MODELS * NUM_CLASSES, N_VOLS + 1))
        for j in range(NUM_MODELS):
            a_feature = alpha_feature_pred[j][i]
            b_feature = alpha_smooth_pred[j][i] + alpha_artifact_pred[j][i]
            if isinstance(a_feature, np.ndarray) and isinstance(b_feature, np.ndarray):
                for l in range(NUM_CLASSES):
                    pmf_feature[j * NUM_CLASSES + l, :] = betabinom.pmf(range(N_VOLS + 1), N_VOLS, a_feature[l], b_feature[l])
            else:
                pmf_feature[j, :] = betabinom.pmf(range(N_VOLS + 1), N_VOLS, a_feature, b_feature)
        mean_pmf_feature = np.mean(pmf_feature, axis=0)
        feature_count = sample_posterior(N_VOLS, mean_pmf_feature, n_samples=N_RUNS)

        for k in range(N_RUNS):
            N_FEATURE = feature_count[k]
            pmf_edgeon = np.zeros((NUM_MODELS * NUM_CLASSES, N_FEATURE + 1))
            for j in range(NUM_MODELS):
                a_disk = alpha_edgeon_pred[j][i]
                b_disk = alpha_else_pred[j][i]
                if isinstance(a_disk, np.ndarray) and isinstance(b_disk, np.ndarray):
                    for l in range(NUM_CLASSES):
                        pmf_edgeon[j * NUM_CLASSES + l, :] = betabinom.pmf(range(N_FEATURE + 1), N_FEATURE, a_disk[l], b_disk[l])
                else:
                    pmf_edgeon[j, :] = betabinom.pmf(range(N_FEATURE + 1), N_FEATURE, a_disk, b_disk)
            mean_pmf_edgeon = np.mean(pmf_edgeon, axis=0)
            edgeon_count[k] = sample_posterior(N_FEATURE, mean_pmf_edgeon)

            N_FACEON = feature_count[k] - edgeon_count[k]
            pmf_bar = np.zeros((NUM_MODELS * NUM_CLASSES, N_FACEON + 1))
            for j in range(NUM_MODELS):
                a_bar = alpha_strong_pred[j][i] + alpha_weak_pred[j][i]
                b_bar = alpha_none_pred[j][i]
                if isinstance(a_bar, np.ndarray) and isinstance(b_bar, np.ndarray):
                    for l in range(NUM_CLASSES):
                        pmf_bar[j * NUM_CLASSES + l, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_bar[l], b_bar[l])
                else:
                    pmf_bar[j, :] = betabinom.pmf(range(N_FACEON + 1), N_FACEON, a_bar, b_bar)
            mean_pmf_bar = np.mean(pmf_bar, axis=0)
            bar_count[k] = sample_posterior(N_FACEON, mean_pmf_bar)

        results.append({
            'id': pred.loc[i, 'id_str'],
            'feature_count': np.array2string(feature_count, separator=', '),
            'edgeon_count': np.array2string(edgeon_count, separator=', '),
            'bar_count': np.array2string(bar_count, separator=', ')
        })

        if i % 100 == 0:
            end = time.time()
            print(f'Filter: {FILTER}, Progress: {i}/{len(pred)}, Time elapsed: {end-start}', flush=True)

    # Convert results to DataFrame
    sampling_result = pd.DataFrame(results)

    # Save results to CSV
    output_path = os.path.join(cat_dir, f'bars_COSMOS_{FILTER}_effnet_m27_sampling.csv')
    sampling_result.to_csv(output_path, index=False)

    print(f'Sampling completed for filter: {FILTER}, results saved to {output_path}')
