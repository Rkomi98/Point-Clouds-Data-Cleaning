import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity, KDTree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import laspy
import open3d as o3d
from sklearn.model_selection import StratifiedKFold, ParameterGrid
gc.enable()
from joblib import Parallel, delayed


# --- Helper Functions ---
def build_kdtree(point_cloud, leaf_size=10):
    """Builds a KDTree (sklearn) with optimized leaf size."""
    return KDTree(point_cloud[:, :3], leaf_size=leaf_size, metric='euclidean')


def estimate_density_knn(point_cloud, tree, k=10):
    """Optimized k-NN density estimation with vectorized operations."""
    distances, _ = tree.query(point_cloud[:, :3], k=k + 1)
    volumes = (4 / 3) * np.pi * np.maximum(distances[:, -1] ** 3, 1e-6)
    return k / volumes


def silverman_bandwidth(data):
    """Calculates bandwidth using Silverman's rule."""
    n, d = data.shape
    A = np.minimum(np.std(data, axis=0), np.subtract(*np.percentile(data, [75, 25], axis=0)) / 1.34)
    A = np.where(A < 1e-6, 0.1, A)  # Avoid division by zero
    return np.mean(A) * (n ** (-1 / (d + 4)))


def simple_prior_adjustment(densities, density_threshold):
    """Prior adjustment with vectorized thresholding."""
    return np.where(densities <= density_threshold, 1e-6, 1.0 - 1e-6)


def classify_points_bayesian(X_train, X_test, tree, radius, prior_probs_train, bandwidth, kernel='gaussian',
                             min_likelihood=1e-6, likelihood_noise=1e-4, n_jobs=-1):
    """Parallelized Bayesian classification of points."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(classify_point)(X_train, point, tree, radius, prior_probs_train[i], bandwidth, kernel, min_likelihood, likelihood_noise)
        for i, point in enumerate(X_test)
    )
    return np.array(results)


def process_fold(fold_data):
    """Processes a single fold (for parallelization)."""
    X_train, X_test, y_test, radius, kernel, density_threshold_percentile, k, min_likelihood, likelihood_noise, df_merged, test_index, fold, output_dir, tree = fold_data
    
    start_time = time.time()
    bandwidth = silverman_bandwidth(X_train)  # Precompute bandwidth
    densities = estimate_density_knn(X_train, tree, k=k)
    density_threshold = np.percentile(densities, density_threshold_percentile)
    prior_probs_train = simple_prior_adjustment(densities, density_threshold)
    y_pred = classify_points_bayesian(X_train, X_test, tree, radius, prior_probs_train, bandwidth, kernel, min_likelihood, likelihood_noise)
    
    execution_time = time.time() - start_time
    return indices_confusion_matrix(y_test, y_pred, f'Bayesian_r_{radius}_fold_{fold + 1}', execution_time=execution_time)


def run_bayesian_classification_and_evaluate(point_cloud, df_merged, radius_values, kernel='gaussian', output_dir='output', cv_folds=5, k_values=[15], density_threshold_percentile_values=[50], min_likelihood_values=[1e-4], likelihood_noise_values=[1e-4], n_jobs=-1):
    """Optimized training loop with Grid Search, and parallelized cross-validation."""
    point_cloud = np.asarray(point_cloud, dtype=np.float32)  # Ensure correct dtype
    tree = build_kdtree(point_cloud)  # Build KDTree once
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_grid = {'radius': radius_values, 'k': k_values, 'density_threshold_percentile': density_threshold_percentile_values, 'min_likelihood': min_likelihood_values, 'likelihood_noise': likelihood_noise_values}
    
    fold_data_list = [
        (point_cloud[train_index], point_cloud[test_index], y_true[test_index], params['radius'], kernel, params['density_threshold_percentile'], params['k'], params['min_likelihood'], params['likelihood_noise'], df_merged, test_index, fold, output_dir, tree)
        for fold, (train_index, test_index) in enumerate(skf.split(point_cloud, y_true))
        for params in ParameterGrid(param_grid)
    ]
    
    results = Parallel(n_jobs=n_jobs)(delayed(process_fold)(data) for data in fold_data_list)
    return results


def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    num_noisy_points = int(noise_percentage / 100 * df.shape[0])
    indices = np.random.choice(df.index, num_noisy_points, replace=False)
    noisy_points = df.loc[indices].copy()
    noisy_points[['X', 'Y', 'Z']] += np.random.normal(0, position_noise_std, (num_noisy_points, 3))
    for color in ['Red', 'Green', 'Blue']:
        if color in df.columns:
            noisy_points[color] = np.clip(noisy_points[color] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1)
    noisy_points['Tag'] = 'Noise'
    return pd.concat([df, noisy_points], ignore_index=True)


def add_outliers_to_dataframe(df, num_outlier_clusters=4, cluster_size_range=(50, 200), cluster_distance_range=(1, 4), position_noise_std=5.0):
    outlier_points = []
    for _ in range(num_outlier_clusters):
        cluster_center = np.random.uniform(df[['X', 'Y', 'Z']].min().values - cluster_distance_range[1], df[['X', 'Y', 'Z']].max().values + cluster_distance_range[1], size=3)
        cluster_size = np.random.randint(*cluster_size_range)
        cluster_points = np.random.normal(loc=cluster_center, scale=position_noise_std, size=(cluster_size, 3))
        outlier_df = pd.DataFrame(cluster_points, columns=['X', 'Y', 'Z'])
        outlier_df['Tag'] = 'Outlier'
        outlier_df[['Red', 'Green', 'Blue']] = np.random.uniform(0, 1, (cluster_size, 3))
        outlier_points.append(outlier_df)
    return pd.concat([df] + outlier_points, ignore_index=True)


def main():
    las_file = laspy.read("/data/landfills_UAV/3dData/FinalMesh/Asbestos45.las")
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    if 'red' in las_file.point_format.dimension_names:
        df['Red'], df['Green'], df['Blue'] = las_file.red / 65535.0, las_file.green / 65535.0, las_file.blue / 65535.0
    if fmt.dimension_by_name('tag'):
        df['Tag'] = np.vectorize({1:'Vegetation', 2:'Terrain', 3:'Metals', 4:'Asbestos', 5:'Tyres', 6:'Plastics'}.get)(las_file.tag)
    else:
        df['Tag'] = 'default'
    
    df_merged = add_noise_to_dataframe(df)
    df_merged = add_outliers_to_dataframe(df_merged)
    point_cloud = df_merged[['X', 'Y', 'Z']].to_numpy()

    results = run_bayesian_classification_and_evaluate(
        point_cloud, df_merged, radius_values=[1.0, 1.5], kernel='gaussian', output_dir='output',
        cv_folds=5, k_values=[10, 20, 30], density_threshold_percentile_values=[25, 50, 75],
        min_likelihood_values=[1e-6, 1e-4], likelihood_noise_values=[1e-3, 1e-4, 1e-5], n_jobs=-1
    )
    
    best_result = max(results, key=lambda x: x['f1'])
    print("\nBest Result:", best_result)
    print("\nAll Results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()