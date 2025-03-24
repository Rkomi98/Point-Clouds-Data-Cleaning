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
import gc
from joblib import Parallel, delayed


# --- Helper Functions ---
def build_kdtree(point_cloud):
    """Builds a KDTree (sklearn)."""
    return KDTree(point_cloud[:, :3], leaf_size=30, metric='euclidean')


# --- Core Algorithm Functions ---

def estimate_density_knn(point_cloud, tree, k=10):
    """Optimized k-NN density estimation."""
    distances, _ = tree.query(point_cloud[:, :3], k=k + 1)
    volumes = (4 / 3) * np.pi * np.clip(distances[:, -1] ** 3, a_min=1e-6, a_max=None)
    return k / volumes


def silverman_bandwidth(data):
    """Calculates bandwidth using Silverman's rule."""
    n, d = data.shape
    if d != 3:
        raise ValueError("Data must be 3D (X, Y, Z)")
    std_dev = np.std(data, axis=0)
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    A = np.minimum(std_dev, iqr / 1.349)
    A = np.where(A < 1e-6, 0.1, A)
    return np.mean(A) * (n ** (-1 / (d + 4)))


def simple_prior_adjustment(densities, density_threshold):
    """Simpler prior adjustment: 0 if below threshold, 1 if above."""
    return np.where(densities <= density_threshold, 1e-6, 1.0 - 1e-6)


def adaptive_kde(X_train, neighbors_train, test_point, bandwidth, kernel='gaussian'):
    """Adaptive KDE: fits on training neighbors, evaluates test point."""
    if len(neighbors_train) < 2:
        return 0.0

    kde = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', leaf_size=30, metric='euclidean',
                        kernel=kernel)
    kde.fit(X_train[neighbors_train, :3])
    try:
        log_likelihood = kde.score_samples(test_point.reshape(1, -1))
        return np.exp(log_likelihood)[0]
    except (ValueError, np.linalg.LinAlgError):
        return 0.0


def classify_point(X_train, point, tree, radius, prior_prob, bandwidth, kernel, min_likelihood, likelihood_noise):
    """Classifies a single point."""
    neighbors_train = tree.query_radius([point], r=radius)[0]

    if len(neighbors_train) < 2:
        return 0

    likelihood_real = adaptive_kde(X_train, neighbors_train, point, bandwidth, kernel)
    likelihood_real = max(likelihood_real, min_likelihood)
    prior_real = 1.0 - prior_prob
    prior_noise = prior_prob

    denominator = likelihood_real * prior_real + likelihood_noise * prior_noise
    if denominator > 0:
        posterior_real = (likelihood_real * prior_real) / denominator
        return 0 if posterior_real > 0.5 else 1
    else:
        return 0


def classify_points_bayesian(X_train, X_test, tree, radius, prior_probs_train, bandwidth, kernel='gaussian',
                             min_likelihood=1e-6, likelihood_noise=1e-4, n_jobs=-1):
    """Classifies points in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(classify_point)(X_train, X_test[i], tree, radius, prior_probs_train[i],
                               bandwidth, kernel, min_likelihood, likelihood_noise)
        for i in range(X_test.shape[0])  # Correctly iterate over indices
    )
    return np.array(results)



# --- Evaluation and Output ---

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time=0, original_tags=None, output_dir=None,
                             threshold=None, sor_params=None):
    """Generates a confusion matrix and metrics."""
    label_mapping = {0: 'Inliers', 1: 'Outlier'}
    y_true_str = [label_mapping[label] for label in y_true]
    y_pred_str = [label_mapping[label] for label in y_pred]

    conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=['Inliers', 'Outlier'])
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    accuracy = accuracy_score(y_true_str, y_pred_str)
    precision = precision_score(y_true_str, y_pred_str, pos_label='Outlier', zero_division=0)
    recall = recall_score(y_true_str, y_pred_str, pos_label='Outlier', zero_division=0)
    f1 = f1_score(y_true_str, y_pred_str, pos_label='Outlier', zero_division=0)

    misclassified_mask = (y_true != y_pred)
    if original_tags is not None:
        if isinstance(original_tags, pd.Series):
            misclassified_tags = original_tags[misclassified_mask].value_counts()
        else:
            misclassified_tags = pd.Series(original_tags[misclassified_mask]).value_counts()
    else:
        misclassified_tags = pd.Series(dtype=int)

    threshold_str = f"(threshold={threshold})" if threshold is not None else ""
    sor_param_str = f"(nb_neighbors={sor_params['nb_neighbors']}, std_ratio={sor_params['std_ratio']})" if sor_params else ""

    print(f"\n--- Metrics for {method_name} {threshold_str} {sor_param_str}---")
    print("\nMisclassified Tag Distribution:")
    print(misclassified_tags)

    if output_dir:
        metrics_filename = os.path.join(output_dir, f"{method_name}{threshold_str}{sor_param_str}_metrics.txt")
        with open(metrics_filename, "w") as f:
            f.write(f"Metrics for {method_name} {threshold_str} {sor_param_str} (Detecting Outlier/Noise)\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}\n")
            f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
            f.write(f"Execution Time: {execution_time:.4f} seconds\n")
            f.write("\nMisclassified Tag Distribution:\n")
            f.write(misclassified_tags.to_string())

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Outlier/Noise', 'Outlier/Noise'], yticklabels=['Not Outlier/Noise', 'Outlier/Noise'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    title_str = f"Confusion Matrix: {method_name} {threshold_str} {sor_param_str} (Outlier/Noise Detection)"
    plt.title(title_str)

    if output_dir:
        plot_filename = os.path.join(output_dir, f"{method_name}{threshold_str}{sor_param_str}_confusion_matrix.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1


def process_fold(fold_data):
    """Processes a single fold with better error handling and memory management."""
    try:
        X_train, X_test, y_test, radius, kernel, density_threshold_percentile, k, min_likelihood, likelihood_noise, df_merged, test_index, fold, output_dir, tree = fold_data

        start_time = time.time()

        # Silverman bandwidth calculation
        bandwidth = silverman_bandwidth(X_train)

        # Density estimation and prior calculation with memory cleanup
        densities = estimate_density_knn(X_train, tree, k=k)
        density_threshold = np.percentile(densities, density_threshold_percentile)
        prior_probs_train = simple_prior_adjustment(densities, density_threshold)
        
        # Clear memory
        del densities
        gc.collect()

        # Break down classification into smaller chunks
        chunk_size = 10000  # Adjust based on your available memory
        n_chunks = int(np.ceil(X_test.shape[0] / chunk_size))
        y_pred = np.zeros(X_test.shape[0], dtype=np.int32)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, X_test.shape[0])
            X_test_chunk = X_test[start_idx:end_idx]
            prior_probs_chunk = prior_probs_train[start_idx:end_idx]
            
            chunk_results = classify_points_bayesian(
                X_train, X_test_chunk, tree, radius, prior_probs_chunk,
                bandwidth, kernel, min_likelihood, likelihood_noise,
                n_jobs=1  # Use single thread within chunks
            )
            y_pred[start_idx:end_idx] = chunk_results

        end_time = time.time()
        execution_time = end_time - start_time
        
        original_tags_test = df_merged['Tag'].iloc[test_index]
        print(f"Fold {fold+1} completed. Params: radius={radius}, k={k}, threshold={density_threshold_percentile}")
        print(f"Execution time: {execution_time:.2f} seconds")

        return indices_confusion_matrix(
            y_test, y_pred,
            f'Bayesian_r_{radius}_fold_{fold + 1}',
            execution_time=execution_time,
            original_tags=original_tags_test,
            output_dir=output_dir
        )
        
    except Exception as e:
        print(f"Error in fold {fold}: {str(e)}")
        return None

def run_bayesian_classification_and_evaluate(point_cloud, df_merged, radius_values,
                                           kernel='gaussian', output_dir='output', cv_folds=5,
                                           k_values=[15], density_threshold_percentile_values=[50],
                                           min_likelihood_values=[1e-4], likelihood_noise_values=[1e-4],
                                           n_jobs=-1):
    """Modified training loop with better memory management and error handling."""
    
    # Convert to float32 for memory efficiency
    if not isinstance(point_cloud, np.ndarray) or point_cloud.dtype != np.float32:
        point_cloud = np.array(point_cloud, dtype=np.float32)
    gc.collect()

    results = []
    inlier_mask = ~df_merged['Tag'].isin(['Outlier', 'Noise'])
    X_inliers = point_cloud[inlier_mask]
    X_outliers = point_cloud[~inlier_mask]
    y_true = np.hstack([np.zeros(X_inliers.shape[0]), np.ones(X_outliers.shape[0])])
    X_combined = np.vstack([X_inliers, X_outliers])
    
    # Use a smaller number of jobs if the dataset is large
    actual_n_jobs = min(n_jobs if n_jobs > 0 else os.cpu_count(), 4)
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    tree = build_kdtree(X_inliers)

    param_grid = {
        'radius': radius_values,
        'k': k_values,
        'density_threshold_percentile': density_threshold_percentile_values,
        'min_likelihood': min_likelihood_values,
        'likelihood_noise': likelihood_noise_values
    }

    for params in ParameterGrid(param_grid):
        radius, k, density_threshold_percentile, min_likelihood, likelihood_noise = (
            params['radius'], params['k'], params['density_threshold_percentile'],
            params['min_likelihood'], params['likelihood_noise']
        )

        print(f"\nEvaluating parameters: {params}")
        
        fold_data_list = []
        for fold, (train_index, test_index) in enumerate(skf.split(X_combined, y_true)):
            X_train, X_test = X_inliers, X_combined[test_index]
            y_test = y_true[test_index]

            fold_data = (X_train, X_test, y_test, radius, kernel, density_threshold_percentile,
                        k, min_likelihood, likelihood_noise, df_merged, test_index, fold, output_dir, tree)
            fold_data_list.append(fold_data)
        print('Before Parallel')
        # Process folds with better error handling
        with Parallel(n_jobs=actual_n_jobs, verbose=10) as parallel:
            fold_results = parallel(delayed(process_fold)(data) for data in fold_data_list)
        
        print('After Parallel')

        # Filter out None results from failed folds
        fold_results = [r for r in fold_results if r is not None]
        
        if fold_results:
            avg_results = {
                'accuracy': np.mean([res[0] for res in fold_results]),
                'precision': np.mean([res[1] for res in fold_results]),
                'recall': np.mean([res[2] for res in fold_results]),
                'f1': np.mean([res[3] for res in fold_results])
            }
            avg_results.update(params)
            avg_results['kernel'] = kernel
            avg_results['bandwidth_method'] = 'silverman'
            results.append(avg_results)
        
        # Force garbage collection between parameter sets
        gc.collect()
    return results

# --- Data Loading and Preprocessing ---
output_dir = "Bayesian_Results/Gemini"
os.makedirs(output_dir, exist_ok=True)

# --- Load LAS Data ---
las_file = laspy.read("/data/landfills_UAV/3dData/FinalMesh/Asbestos45.las")
points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
fmt = las_file.point_format

if 'red' in las_file.point_format.dimension_names:
    red = las_file.red / 65535.0
    green = las_file.green / 65535.0
    blue = las_file.blue / 65535.0
    colors = np.vstack((red, green, blue)).transpose()
    print("Colors!")
else:
    colors = None
    print("No Colors!")

if fmt.dimension_by_name('tag'):
    tags_numeric = las_file.tag
    print("Tags!")
else:
    tags_numeric = None
    print("No tags :-(")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)

df = pd.DataFrame(data=points, columns=['X', 'Y', 'Z'])
if colors is not None:
    df['Red'] = red
    df['Green'] = green
    df['Blue'] = blue

if tags_numeric is not None:
    tag_mapping = {
        "Vegetation": 1,
        "Terrain": 2,
        "Metals": 3,
        "Asbestos": 4,
        "Tyres": 5,
        "Plastics": 6,
        "default": 0
    }
    inverse_tag_mapping = {v: k for k, v in tag_mapping.items()}
    vectorized_mapping = np.vectorize(inverse_tag_mapping.get)
    tags = vectorized_mapping(tags_numeric)
    df['Tag'] = tags
else:
    df['Tag'] = 'default'

terrain_points = df[df['Tag'] == 'Terrain']
vegetation_points = df[df['Tag'] == 'Vegetation']

if not terrain_points.empty:
    min_terrain_z = np.min(terrain_points['Z'])
else:
    min_terrain_z = df['Z'].min()

if not vegetation_points.empty:
    max_terrain_z = np.max(vegetation_points['Z'])
else:
    max_terrain_z = df['Z'].max()
df = df[(df['Z'] >= min_terrain_z)]


# --- Add Noise and Outliers ---
def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    num_points = df.shape[0]
    num_noisy_points = int(noise_percentage / 100 * num_points)
    indices = np.random.choice(df.index, num_noisy_points, replace=False)
    noisy_points = df.loc[indices, ['X', 'Y', 'Z']].copy()
    noisy_points['X'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Y'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Z'] += np.random.normal(0, position_noise_std, num_noisy_points)
    if 'Red' in df.columns and 'Green' in df.columns and 'Blue' in df.columns:
        noisy_points['Red'] = df.loc[indices, 'Red'] + np.random.normal(0, color_noise_std, num_noisy_points)
        noisy_points['Green'] = df.loc[indices, 'Green'] + np.random.normal(0, color_noise_std, num_noisy_points)
        noisy_points['Blue'] = df.loc[indices, 'Blue'] + np.random.normal(0, color_noise_std, num_noisy_points)
        noisy_points['Red'] = noisy_points['Red'].clip(0, 1)
        noisy_points['Green'] = noisy_points['Green'].clip(0, 1)
        noisy_points['Blue'] = noisy_points['Blue'].clip(0, 1)
    else:
        noisy_points['Red'] = np.nan
        noisy_points['Green'] = np.nan
        noisy_points['Blue'] = np.nan
    noisy_points['Tag'] = 'Noise'
    df_merged = pd.concat([df, noisy_points], ignore_index=True)
    return df_merged


def add_outliers_to_dataframe(df, num_outlier_clusters=4, cluster_size_range=(50, 200),
                              cluster_distance_range=(1, 4), position_noise_std=5.0):
    min_x, max_x = df['X'].min(), df['X'].max()
    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = min_terrain_z, max_terrain_z
    outlier_points = []
    for _ in range(num_outlier_clusters):
        cluster_center = [
            np.random.uniform(min_x - cluster_distance_range[1], max_x + cluster_distance_range[1]),
            np.random.uniform(min_y - cluster_distance_range[1], max_y + cluster_distance_range[1]),
            np.random.uniform(min_z - cluster_distance_range[1], max_z + cluster_distance_range[1])
        ]
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        cluster_points = np.random.normal(
            loc=cluster_center,
            scale=position_noise_std,
            size=(int(cluster_size), 3)
        )
        outlier_df = pd.DataFrame(cluster_points, columns=['X', 'Y', 'Z'])
        outlier_df['Tag'] = 'Outlier'
        outlier_df['Red'] = np.random.uniform(0, 1, cluster_size)
        outlier_df['Green'] = np.random.uniform(0, 1, cluster_size)
        outlier_df['Blue'] = np.random.uniform(0, 0.01, cluster_size)
        outlier_points.append(outlier_df)
    outlier_df_combined = pd.concat(outlier_points, ignore_index=True)
    df_merged = pd.concat([df, outlier_df_combined], ignore_index=True)
    return df_merged


# --- Main Execution ---

if __name__ == "__main__":
    # Parameter grid
    radius_values = [1.5]
    kernel = 'gaussian'
    cv_folds = 10
    k_values = [10,5]
    density_threshold_percentile_values = [5, 50]
    min_likelihood_values = [1e-4] #1e-6,
    likelihood_noise_values = [1e-2] #1e-3, 1e-4,
    n_jobs = -1

    df_merged = add_noise_to_dataframe(df)
    df_merged = add_outliers_to_dataframe(df_merged)
    point_cloud = df_merged[['X', 'Y', 'Z']].to_numpy()

    results = run_bayesian_classification_and_evaluate(
        point_cloud, df_merged, radius_values, kernel, output_dir, cv_folds,
        k_values=k_values, density_threshold_percentile_values=density_threshold_percentile_values,
        min_likelihood_values=min_likelihood_values, likelihood_noise_values=likelihood_noise_values,
        n_jobs=n_jobs
    )

    best_result = max(results, key=lambda x: x['f1'])
    print("\nBest Result:")
    print(best_result)

    print("\nAll Results:")
    for result in results:
        print(result)