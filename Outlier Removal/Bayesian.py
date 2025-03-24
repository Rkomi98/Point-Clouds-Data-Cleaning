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
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

# --- Helper Functions ---
def build_kdtree(point_cloud):
    """Builds a KDTree (sklearn)."""
    return KDTree(point_cloud[:, :3], leaf_size=30, metric='euclidean')

def compute_median(values):
    """Computes the median."""
    return np.median(values)

# --- Core Algorithm Functions ---

def estimate_density_knn(point_cloud, tree, k=10):
    """Optimized k-NN density estimation."""
    distances, _ = tree.query(point_cloud[:, :3], k=k + 1)
    return k / ((4/3) * np.pi * (np.maximum(distances[:, -1], 1e-6) ** 3))

def silverman_bandwidth(data):
    """Calculates bandwidth using Silverman's rule."""
    n, d = data.shape
    if d != 3:
        raise ValueError("Data must be 3D (X, Y, Z)")
    std_dev = np.std(data, axis=0)
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    A = np.minimum(std_dev, iqr / 1.34)
    A[A < 1e-6] = 0.1
    return np.mean(A) * (n ** (-1 / (d + 4)))

def kde_bandwidth_cross_validation(data, bandwidths, cv=5):
    """Cross-validation for bandwidth selection."""
    params = {'bandwidth': bandwidths}
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(KernelDensity(algorithm='kd_tree', leaf_size=30, metric='euclidean',
                                     kernel='gaussian', rtol=0.0, atol=0.0, breadth_first=True,
                                     sort_results=True), params, cv=kfold, verbose=0, n_jobs=-1)
    grid.fit(data)
    return grid.best_params_['bandwidth']

def sigmoid_prior_adjustment(densities, density_threshold, steepness=1):
    """Numerically stable sigmoid for prior adjustment."""
    normalized = densities / density_threshold
    z = steepness * (normalized - 1)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def adaptive_kde(X_train, neighbors_train, test_point, bandwidth, kernel='gaussian'):
    """Adaptive KDE: fits on training neighbors, evaluates test point."""
    if len(neighbors_train) < 2:
        return 0.0

    kde = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', leaf_size=30, metric='euclidean',
                        kernel=kernel, rtol=0.0, atol=0.0, breadth_first=True)
    kde.fit(X_train[neighbors_train, :3])
    log_likelihood = kde.score_samples(test_point.reshape(1, -1))
    return np.exp(log_likelihood)[0]

def classify_points_bayesian(X_train, X_test, tree, radius, prior_probs_train, bandwidth, kernel='gaussian', batch_size=1000, min_likelihood=1e-6):
    """Classifies points in batches."""
    num_points = X_test.shape[0]
    n_batches = (num_points + batch_size - 1) // batch_size
    results = []

    for _ in range(n_batches):
        start = _ * batch_size
        end = min((_ + 1) * batch_size, X_test.shape[0])
        batch_X_test = X_test[start:end]
        batch_results = []

        for i in range(len(batch_X_test)):
            point = batch_X_test[i]
            neighbors_train = tree.query_radius([point], r=radius)[0]

            if len(neighbors_train) < 2:
                batch_results.append(0)
                continue

            likelihood_real = adaptive_kde(X_train, neighbors_train, point, bandwidth, kernel)
            likelihood_real = max(likelihood_real, min_likelihood)
            likelihood_noise = 1e-4
            prior_real = 1.0 - prior_probs_train[start + i]
            prior_noise = prior_probs_train[start + i]

            posterior_real = (likelihood_real * prior_real) / (
                likelihood_real * prior_real + likelihood_noise * prior_noise
            )
            batch_results.append(0 if posterior_real > 0.5 else 1)

        results.extend(batch_results)
    return np.array(results)

# --- Evaluation and Output ---

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time=0, original_tags=None, output_dir=None, threshold=None, sor_params=None):
    """Generates a confusion matrix and metrics (same as before)."""
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

    # Handling misclassified tags when original_tags is a pandas Series
    misclassified_mask = (y_true != y_pred)
    if original_tags is not None:
        if isinstance(original_tags, pd.Series):
            misclassified_tags = original_tags[misclassified_mask].value_counts()
        else:
            misclassified_tags = pd.Series(original_tags[misclassified_mask]).value_counts()
    else:
        misclassified_tags = pd.Series()

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

def run_bayesian_classification_and_evaluate(point_cloud, df_merged, radius_values, bandwidth_method='silverman', kernel='gaussian', output_dir='output', cv_folds=5, k=15, density_threshold_percentile=50, steepness=2, min_likelihood=1e-4):
    """Optimized training loop with parameter tuning."""
    point_cloud = point_cloud.astype(np.float32)
    import gc
    gc.collect()

    results = []
    # Separate inliers (original data) and outliers/noise
    inlier_mask = ~df_merged['Tag'].isin(['Outlier', 'Noise'])
    X_inliers = point_cloud[inlier_mask]
    X_outliers = point_cloud[~inlier_mask]
    y_true = np.hstack([np.zeros(X_inliers.shape[0]), np.ones(X_outliers.shape[0])])
    
    # Combine inliers and outliers for test splits
    X_combined = np.vstack([X_inliers, X_outliers])
    
    # Use StratifiedKFold to maintain class distribution
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for radius in radius_values:
        fold_results = []
        for fold, (train_index, test_index) in enumerate(skf.split(X_combined, y_true)):
            print(f"Evaluating radius={radius}, Fold={fold+1}/{cv_folds}")
            start_time = time.time()

            # Training data: only inliers (class 0)
            X_train = X_inliers  # Use all inliers for training
            # Test data: includes both inliers and outliers from the split
            X_test = X_combined[test_index]
            y_test = y_true[test_index]

            tree = build_kdtree(X_train)
            bandwidth = silverman_bandwidth(X_train.astype(np.float32)) if bandwidth_method == 'silverman' else bandwidth_method
            densities = estimate_density_knn(X_train, tree, k=k).astype(np.float32)
            prior_probs_train = sigmoid_prior_adjustment(densities, np.percentile(densities, density_threshold_percentile), steepness=steepness).astype(np.float32)
            del densities
            gc.collect()

            y_pred = classify_points_bayesian(X_train, X_test, tree, radius, prior_probs_train, bandwidth, kernel, batch_size=500, min_likelihood=min_likelihood)

            end_time = time.time()
            execution_time = end_time - start_time
            accuracy, precision, recall, f1 = indices_confusion_matrix(
                y_test, y_pred,
                f'Bayesian_r_{radius}_fold_{fold+1}',
                execution_time=execution_time,
                original_tags=df_merged.iloc[test_index]['Tag'],  # Adjust indexing if needed
                output_dir=output_dir
            )
            fold_results.append({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

        avg_results = {metric: np.mean([res[metric] for res in fold_results]) for metric in fold_results[0]}
        avg_results['radius'] = radius
        avg_results['k'] = k
        avg_results['density_threshold_percentile'] = density_threshold_percentile
        avg_results['steepness'] = steepness
        avg_results['bandwidth'] = bandwidth
        avg_results['min_likelihood'] = min_likelihood
        results.append(avg_results)

    return results
# --- Data Loading and Preprocessing ---
# --- Output Directories ---
output_dir = "Bayesian_Results/Radius"
os.makedirs(output_dir, exist_ok=True)

# --- Load LAS Data ---
las_file = laspy.read("/data/landfills_UAV/3dData/FinalMesh/Asbestos45.las") # Path
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
    tags = None
    print("No tags :-(")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors) # Add colors to pcd for visualization

df = pd.DataFrame(data=points, columns=['X','Y','Z'])
if colors is not None:
    df['Red'] = red
    df['Green'] = green
    df['Blue'] = blue

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

terrain_points = df[df['Tag'] == 'Terrain']
vegetation_points = df[df['Tag'] == 'Vegetation']
terrain_coords = terrain_points[['X', 'Y', 'Z']].values
veg_coords = vegetation_points[['X', 'Y', 'Z']].values
min_terrain_z = np.min(terrain_coords[:, 2])
max_terrain_z = np.max(veg_coords[:, 2])
df = df[df['Z'] >= min_terrain_z]

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
    # Parameters (Tunable)
    radius_values = [1, 1.5, 2]
    bandwidth_method = 'silverman'
    kernel = 'gaussian'
    #kernel = 'epanechnikov'
    output_dir = output_dir
    cv_folds = 5
    k = 140
    density_threshold_percentile = 12.5
    steepness = 2.0
    min_likelihood = 1e-4

    df_merged = add_noise_to_dataframe(df)
    df_merged = add_outliers_to_dataframe(df_merged)
        # Create y_true *before* splitting
    y_true = np.where(df_merged['Tag'].isin(['Outlier', 'Noise']), 1, 0)

    # Prepare point cloud data
    point_cloud = df_merged[['X', 'Y', 'Z']].to_numpy()

    # Run classification and evaluation
    results = run_bayesian_classification_and_evaluate(
        point_cloud, df_merged, radius_values, bandwidth_method, kernel, output_dir, cv_folds,
        k=k, density_threshold_percentile=density_threshold_percentile, steepness=steepness, min_likelihood=min_likelihood
    )

    # Output results
    for result in results:
        print(f"Radius: {result['radius']}, Bandwidth: {result['bandwidth']}, Kernel: {result['kernel']}")
        print(f"k: {result['k']}, Density Threshold Percentile: {result['density_threshold_percentile']}, Steepness: {result['steepness']}")
        print(f"Min Likelihood: {result['min_likelihood']}")
        print(f"  Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"Recall: {result['recall']:.4f}, F1 Score: {result['f1']:.4f}")