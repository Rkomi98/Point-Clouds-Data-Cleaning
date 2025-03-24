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
from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity

def build_kdtree(point_cloud):
    """
    Build a KD-Tree from the point cloud.
    
    Parameters:
    point_cloud (ndarray): Point cloud data (Nx3 for XYZ).
    
    Returns:
    KDTree: KDTree built from the input point cloud.
    """
    tree = KDTree(point_cloud[:, :3])  # Using only XYZ for spatial indexing
    return tree

def compute_median(values):
    """
    Compute the median of a list of values.
    Parameters:
    values (list or ndarray): A list or array of numerical values.
    
    Returns:
    float: The median of the values.
    """
    return np.median(values)

def dynamic_prior_adjustment(point_cloud, tree, radius, density_threshold):
    """
    Adjust prior probabilities for each point based on local neighborhood density.
    Parameters:
    point_cloud (ndarray): Point cloud data (Nx3 for XYZ).
    tree (KDTree): KDTree built from the point cloud.
    radius (float): Search radius to find neighbors.
    density_threshold (float): Threshold density value to classify regions as sparse or dense. # ADDED
    
    Returns:
    ndarray: Adjusted prior probabilities for each point.
    """
    prior_probs = np.zeros(point_cloud.shape[0])
    densities = []  # Collect all density values to compute the median later # No longer needed here

    for i, point in enumerate(point_cloud):
        # Query neighbors within the radius
        neighbors = tree.query_radius([point[:3]], r=radius)[0]
        density = len(neighbors) / (radius ** 3)  # Simple density estimate

        # Adjust the prior probability based on the local density
        if density < density_threshold:  # Use density_threshold instead of some_threshold
            prior_probs[i] = 0.9  # High probability of being noise in sparse regions
        else:
            prior_probs[i] = 0.1  # Low probability of being noise in dense regions

    return prior_probs

def adaptive_kde(point_cloud, neighbors, bandwidth=1.0):
    """
    Perform adaptive KDE for a given point and its neighbors.
    
    Parameters:
    point_cloud (ndarray): Point cloud data.
    neighbors (ndarray): Indices of neighboring points.
    bandwidth (float): Bandwidth for the KDE. If it is too small, noise is not removed, else too high, also important data is deleted
    
    Returns:
    float: The KDE-based likelihood for the point.
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(point_cloud[neighbors, :3])
    
    # Compute the log-likelihood for each point in its neighborhood
    log_densities = kde.score_samples(point_cloud[neighbors, :3])
    
    # Convert log densities to probabilities
    return np.exp(log_densities).mean()  # Average likelihood across neighbors
def classify_points_bayesian(point_cloud, tree, radius, prior_probs, bandwidth):
    """
    Classify points as noise or valid based on Bayesian decision rule.
    
    Parameters:
    point_cloud (ndarray): Point cloud data.
    tree (KDTree): KDTree built from the point cloud.
    radius (float): Radius for neighbor search.
    prior_probs (ndarray): Prior probabilities for each point.
    bandwidth (float): Bandwidth for KDE.
    
    Returns:
    ndarray: Array of classifications (1 for valid, 0 for noise).
    """
    classifications = np.zeros(point_cloud.shape[0])
    
    for i, point in enumerate(point_cloud):
        neighbors = tree.query_radius([point[:3]], r=radius)[0]
        
        # Likelihood from KDE
        likelihood_real = adaptive_kde(point_cloud, neighbors, bandwidth)
        likelihood_noise = 1 - likelihood_real  # Assume complementary likelihood

        # Compute posterior probabilities
        posterior_real = (likelihood_real * (1 - prior_probs[i])) / \
                         (likelihood_real * (1 - prior_probs[i]) + likelihood_noise * prior_probs[i])
        
        # Classify point based on posterior probability
        classifications[i] = 1 if posterior_real > 0.5 else 0
    
    return classifications


# --- Data Loading and Preprocessing ---
output_dir = "Bayesian_Results/OldCode"
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

# Load or generate point cloud data
# Nx3 or Nx6 (XYZRGB), assuming Nx3 for this example
densities = []
df_merged = add_noise_to_dataframe(df)
df_merged = add_outliers_to_dataframe(df_merged)
point_cloud = df_merged[['X', 'Y', 'Z']].to_numpy()
#point_cloud = df_merged[['X', 'Y', 'Z', 'Red', 'Green', 'Blue']].to_numpy()
tree = build_kdtree(point_cloud)

radius = 0.1  # Example search radius -- it is needed for an approximate of the threshold
bandwidth = 0.20  # Bandwidth for KDE

# Step 1: Collect densities for all points
for i, point in enumerate(point_cloud):
    # Query neighbors within the radius
    neighbors = tree.query_radius([point[:3]], r=radius)[0]
    density = len(neighbors) / (radius ** 3)  # Simple density estimate
    densities.append(density)  # Store density

# Step 2: Compute the median density
some_threshold = compute_median(densities)
print(f"Computed median density threshold: {some_threshold}")

# Parameters
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
#some_threshold = 4000  # Density threshold for adjusting priors
#Computed median density threshold: 4999.999999999999
def run_bayesian_classification_and_evaluate(point_cloud, tree, y_true, radius_values, bandwidth):    
    results = []
    # Loop over different radius and bandwidth values
    for radius in radius_values:
        # Step 1: Compute prior probabilities based on density estimation
        prior_probs = dynamic_prior_adjustment(point_cloud, tree, radius, some_threshold)

        print(f"Evaluating with radius={radius}, bandwidth={bandwidth}")

        # Step 2: Classify points using Bayesian method
        y_pred = classify_points_bayesian(point_cloud, tree, radius, prior_probs, bandwidth)           

        # Step 3: Evaluate the performance using confusion matrix metrics
        tpr, fpr = indices_confusion_matrix(y_true, y_pred, method_name=f'Radius={radius}, Bandwidth={bandwidth}') 

        # Step 4: Identify misclassified points (where y_true and y_pred do not match)
        misclassified_points = np.where(y_true != y_pred)[0]  # Indices where true and predicted labels differ

        # Step 5: Get the corresponding ground truth tags for misclassified points
        misclassified_tags = df_merged.iloc[misclassified_points]['Tag'] if misclassified_points.size > 0 else []

        # Step 6: Output the unique ground truth tags for the misclassified points
        unique_misclassified_tags = misclassified_tags.unique() if len(misclassified_tags) > 0 else []

        print("Tags of points classified as Outliers/Noise but are not:")
        print(unique_misclassified_tags)

        # Optionally, you can also output the count of each misclassified tag
        misclassified_tag_counts = misclassified_tags.value_counts() if len(misclassified_tags) > 0 else pd.Series()
        print("\nCounts of misclassified tags:")
        print(misclassified_tag_counts)

        # Step 7: Append the results for this radius and bandwidth combination
        results.append((radius, bandwidth, tpr, fpr))

    return results

# Example usage:
true_labels = np.where(df_merged['Tag'].isin(['Outlier', 'Noise']), 1, 0)
radius_values = [0.4, 0.8, 1.6]  # Example search radius values
#bandwidth_values = [0.2, 0.5, 0.8]  # Example bandwidth values for KDE 0,1, 0.2, 0.3 are too near -- not so important, it is enough 0.2

# Assuming `point_cloud`, `tree`, and `true_labels` (ground truth) are already defined
results = run_bayesian_classification_and_evaluate(point_cloud, tree, true_labels, radius_values, bandwidth)
