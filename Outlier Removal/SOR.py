import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import time
from sklearn.preprocessing import StandardScaler
import os
from scipy import stats
from sklearn.decomposition import PCA
import copy

# --- Output Directories ---
output_dir_sor = "SOR_Results_2_Noise"
os.makedirs(output_dir_sor, exist_ok=True)

# --- Load LAS Data ---
las_file = laspy.read("/data/landfills_UAV/3dData/FinalMesh/Asbestos2.las")
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

df_merged = add_noise_to_dataframe(df)
df_merged = add_outliers_to_dataframe(df_merged)
y_true_original = np.where((df_merged['Tag'] == 'Outlier'), 1, 0) #| (df_merged['Tag'] == 'Noise')

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None, sor_params=None): # MODIFICATION: Added sor_params for SOR info
    label_mapping = {0: 'Inliers', 1: 'Outlier'}
    y_true_str = [label_mapping[label] for label in y_true]
    y_pred_str = [label_mapping[label] for label in y_pred]

    conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=['Inliers', 'Outlier'])
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    accuracy = accuracy_score(y_true_str, y_pred_str)
    precision = precision_score(y_true_str, y_pred_str, pos_label='Outlier')
    recall = recall_score(y_true_str, y_pred_str, pos_label='Outlier')
    f1 = f1_score(y_true_str, y_pred_str, pos_label='Outlier')

    misclassified_mask = (y_true != y_pred)
    misclassified_tags = original_tags[misclassified_mask]
    tag_counts = pd.Series(misclassified_tags).value_counts()

    threshold_str = f"(threshold={threshold})" if threshold is not None else ""
    sor_param_str = f"(nb_neighbors={sor_params['nb_neighbors']}, std_ratio={sor_params['std_ratio']})" if sor_params else "" # MODIFICATION: Added SOR params string

    print(f"\n--- Metrics for {method_name} {threshold_str} {sor_param_str}---") # MODIFICATION: Added sor_param_str to print
    print("\nMisclassified Tag Distribution:")
    print(tag_counts)

    metrics_filename = os.path.join(output_dir, f"{method_name}{threshold_str}{sor_param_str}_metrics_noise_2.txt") # MODIFICATION: Added sor_param_str to filename
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {threshold_str} {sor_param_str} (Detecting Outlier/Noise)\n") # MODIFICATION: Added sor_param_str to header
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
        f.write(tag_counts.to_string())

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
    title_str = f"Confusion Matrix: {method_name} {threshold_str} {sor_param_str} (Outlier/Noise Detection)" # MODIFICATION: Added sor_param_str to title
    plt.title(title_str)

    plot_filename = os.path.join(output_dir, f"{method_name}{threshold_str}{sor_param_str}_confusion_matrix_noise_2.png") # MODIFICATION: Added sor_param_str to filename
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1
# --- SOR Outlier Removal Evaluation Function ---
def evaluate_sor_outlier_removal(pcd_data, y_true, df_original_tags, nb_neighbors, std_ratio, output_dir):
    """
    Evaluates Statistical Outlier Removal (SOR) for outlier detection (CORRECTED).

    Parameters:
    - pcd_data: Open3D PointCloud object.
    - y_true: Ground truth labels (numpy array).
    - df_original_tags: DataFrame column containing original tags for misclassification analysis.
    - nb_neighbors: Number of neighbors for SOR.
    - std_ratio: Standard deviation ratio for SOR.
    - output_dir: Directory to save results.

    Returns:
    - accuracy, precision, recall, f1: Performance metrics.
    """
    start_time_sor = time.time()
    pcd_copy = copy.deepcopy(pcd_data) # Deep copy inside function for clarity
    cl, ind = pcd_copy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    outlier_indices_sor = np.setdiff1d(np.arange(len(pcd_data.points)), np.array(ind)) # Get outlier indices

    # Create y_pred for SOR (for ORIGINAL point cloud - mark outliers as 1)
    predicted_labels_sor = np.zeros(len(pcd_data.points), dtype=int) # Initialize with all 0s
    predicted_labels_sor[outlier_indices_sor] = 1 # Set outliers to 1

    end_time_sor = time.time()
    execution_time_sor = end_time_sor - start_time_sor
    method_name_sor = 'SOR Outlier Removal'

    sor_params = {'nb_neighbors': nb_neighbors, 'std_ratio': std_ratio} # Store params for metrics function
    accuracy_sor, precision_sor, recall_sor, f1_sor = indices_confusion_matrix(
        y_true, # Use ORIGINAL y_true
        predicted_labels_sor,
        method_name_sor,
        execution_time_sor,
        df_original_tags,
        output_dir,
        sor_params=sor_params # Pass sor_params
    )
    return accuracy_sor, precision_sor, recall_sor, f1_sor

# --- Parameter Lists for SOR Evaluation ---
nb_neighbors_list = [15, 20, 25, 30, 35]
#nb_neighbors_list = [50, 100, 150]
#std_ratio_list = [0.001, 0.01, 0.1, 1.0, 3.0, 3.5]
std_ratio_list = [6, 6.5, 7, 7.5, 8.0, 8.5]
sor_results = [] # List to store SOR results
# Step 5: Convert the points and colors to Open3D PointCloud format
pcd = o3d.geometry.PointCloud()
# Extract point coordinates and color data from the DataFrame
point_noise = df_merged[['X', 'Y', 'Z']].to_numpy()  # Convert to NumPy array
color_noise = df_merged[['Red', 'Green', 'Blue']].to_numpy()  # Convert to NumPy array

# Convert the points and colors to Open3D PointCloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_noise)

# Add colors to the point cloud if they exist
if color_noise.size > 0:  # Ensure color_noise is not empty
    pcd.colors = o3d.utility.Vector3dVector(color_noise)
pcd.points = o3d.utility.Vector3dVector(point_noise)

# --- Evaluate SOR for different parameter combinations ---
for nb_neighbors in nb_neighbors_list:
    for std_ratio in std_ratio_list:
        pcd_copy = copy.deepcopy(pcd) # Create deep copy *before* removing outliers
        start_time_sor = time.time()
        cl, ind = pcd_copy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        end_time_sor = time.time()
        execution_time_sor = end_time_sor - start_time_sor
        # Create y_pred for SOR (for ORIGINAL point cloud - mark outliers as 1)
        predicted_labels_sor = np.ones(len(pcd.points), dtype=int) # Initialize with all 0s
        predicted_labels_sor[ind] = 0 # Set inliers to 0

        accuracy_sor, precision_sor, recall_sor, f1_sor = indices_confusion_matrix(
            y_true_original, # Use original y_true
            predicted_labels_sor,
            'SOR Outlier Removal',
            execution_time_sor, 
            df_merged['Tag'], # Use full tags
            output_dir_sor,
            sor_params={'nb_neighbors': nb_neighbors, 'std_ratio': std_ratio}
        )
        sor_results.append({ # Store results
            'nb_neighbors': nb_neighbors,
            'std_ratio': std_ratio,
            'accuracy': accuracy_sor,
            'precision': precision_sor,
            'recall': recall_sor,
            'f1_score': f1_sor
        })


print(f"\nResults for SOR Outlier Removal saved in '{output_dir_sor}' folder.")

# --- Print SOR Comparison Summary ---
print("\n--- SOR Parameter Comparison Summary ---")
for result in sor_results:
    print(f"nb_neighbors: {result['nb_neighbors']}, std_ratio: {result['std_ratio']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print("-" * 40)