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
from scipy.special import erf
from sklearn.decomposition import PCA
import copy
from sklearn.neighbors import NearestNeighbors # Import NearestNeighbors for LoOP
from scipy.stats import norm # Import norm for LoOP probability calculation
import plotly.express as px

# --- Output Directories ---
output_dir_loop = "LoOP_Results_2" # Output directory for LoOP
os.makedirs(output_dir_loop, exist_ok=True)

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
    min_z, max_z = min_terrain_z, max_terrain_z # Corrected max_z to max_terrain_z
    outlier_points = []
    for _ in range(num_outlier_clusters):
        cluster_center = [
            np.random.uniform(min_x - cluster_distance_range[1], max_x + cluster_distance_range[1]),
            np.random.uniform(min_y - cluster_distance_range[1], max_y + cluster_distance_range[1]),
            np.random.uniform(min_z - cluster_distance_range[1], max_terrain_z + cluster_distance_range[1]) # Corrected max_z to max_terrain_z
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
# Corrected line: Use parentheses for each condition and then bitwise OR
y_true_original = np.where((df_merged['Tag'] == 'Outlier') | (df_merged['Tag'] == 'Noise'), 1, 0)

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None, loop_params=None): # Changed loci_params to loop_params
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

    # Calculate total counts for each original tag
    original_tag_counts = pd.Series(original_tags).value_counts()
    misclassified_tag_percentages = {}
    for tag, count in tag_counts.items():
        total_tag_count = original_tag_counts.get(tag, 0)
        percentage = (count / total_tag_count) * 100 if total_tag_count > 0 else 0
        misclassified_tag_percentages[tag] = percentage

    threshold_str = f"(threshold={threshold})" if threshold is not None else ""
    loop_param_str = f"(k={loop_params['k']}, lambda={loop_params['lambda_param']})" if loop_params else "" # Changed loci_params to loop_params,  and params string
    params_str_for_filename = f"{threshold_str}{loop_param_str}".replace("(", "").replace(")", "").replace(",", "_").replace("=", "_").replace(".", "p") # Create filename string

    print(f"\n--- Metrics for {method_name} {params_str_for_filename} ---") # Changed method_name to LoOP
    print("\nMisclassified Tag Distribution:")
    for tag, count in tag_counts.items():
        percentage = misclassified_tag_percentages[tag]
        print(f"  {tag}: {count} points ({percentage:.2f}%)")

    metrics_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_metrics_noise_45.txt") # Changed method_name to LoOP
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {params_str_for_filename} (Detecting Outlier/Noise)\n") # Changed method_name to LoOP
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
        for tag, count in tag_counts.items():
            percentage = misclassified_tag_percentages[tag]
            f.write(f"  {tag}: {count} points ({percentage:.2f}%)\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Outlier/Noise', 'Outlier/Noise'], yticklabels=['Not Outlier/Noise', 'Outlier/Noise'])
    title_str = f"Confusion Matrix: {method_name} {params_str_for_filename} (Outlier/Noise Detection)" # Changed method_name to LoOP
    plt.title(title_str)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plot_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_confusion_matrix_noise_45.png") # Changed method_name to LoOP
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1

# Step 1: Compute probabilistic distance (p-dist)
def compute_p_dist(distances, sigma):
    return np.exp(-np.square(distances) / (2 * sigma ** 2)) # Gaussian kernel distance

# Step 2: Compute local density estimation
def compute_local_density(distances, sigma):
    p_dists = compute_p_dist(distances, sigma)
    local_density = np.mean(p_dists)
    return local_density

# Step 3: LoOP core function
# --- Enhanced LoOP Implementation ---
'''
def loop_outlier_detection(data, k=20, lambda_param=3):
    """
    Correct implementation of LoOP algorithm based on:
    Kriegel, H. P., et al. (2009). LoOP: Local Outlier Probabilities.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)  # +1 to include self
    distances, indices = nbrs.kneighbors(data)
    
    # Remove self distance (0th element)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Compute probabilistic distances (pdist)
    epdist = np.zeros(len(data))
    for i in range(len(data)):
        sigma_i = np.std(distances[i])
        sigma_i = sigma_i if sigma_i > 1e-6 else 1e-6
        p_dists = np.exp(-(distances[i]**2) / (2 * sigma_i**2))
        epdist[i] = np.mean(p_dists)
    
    # Global expected pdist
    E_epdist = np.mean(epdist)
    
    # Compute PLOF (Probabilistic Local Outlier Factor)
    plof = (epdist / E_epdist) - 1
    
    # Compute nPLOF (normalization factor)
    nplof = np.sqrt(np.mean(plof**2)) * np.sqrt(2)
    
    # Handle division by zero
    nplof = nplof if nplof > 1e-6 else 1e-6
    
    # Calculate LoOP scores
    loop_scores = plof / (lambda_param * nplof)
    
    # Convert to probabilities using error function
    probabilities = np.maximum(0, 0.5 * (1 + erf(loop_scores / np.sqrt(2))))
    
    return probabilities
'''
def corrected_loop(data, k=50, lambda_param=3): ## TODO Guardare differenze tra questa e quella sopra
    """Proper LoOP implementation following Kriegel et al."""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Remove self-reference
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # 1. Compute probabilistic distances
    sigma = np.std(distances, axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1e-6, sigma)  # Prevent division by zero
    p_dists = np.exp(-(distances**2)/(2 * sigma**2))
    
    # 2. Local reachability densities
    lrd = 1 / (np.mean(np.sqrt(p_dists), axis=1) + 1e-6)
    
    # 3. Compute PLOF (Probabilistic Local Outlier Factor)
    plof = lrd / np.mean(lrd) - 1
    
    # 4. Normalize using nPLOF
    nplof = np.sqrt(np.mean(plof**2)) * np.sqrt(2)
    nplof = np.clip(nplof, 1e-6, None)
    
    # 5. Final LoOP scores
    loop_scores = plof / (lambda_param * nplof)
    probabilities = 0.5 * (1 + erf(loop_scores / np.sqrt(2)))
    
    return probabilities

# --- LoOP Outlier Detection Evaluation Function ---
def evaluate_loop_outlier_detection(X, y_true, df_original_tags, k, lambda_param, output_dir):
    """
    Evaluates LoOP for outlier detection.
    """
    start_time_loop = time.time()
    #outlier_probabilities_loop = loop_outlier_detection(X, k=k, lambda_param=lambda_param)
    outlier_probabilities_loop = corrected_loop(X, k=k, lambda_param=lambda_param)
    # Convert outlier probabilities to binary labels (threshold 0.5)
    y_pred_loop = np.where(outlier_probabilities_loop > 0.5, 1, 0)
    end_time_loop = time.time()
    execution_time_loop = end_time_loop - start_time_loop
    method_name_loop = 'LoOP Outlier Detection'

    loop_params = {'k': k, 'lambda_param': lambda_param}
    accuracy_loop, precision_loop, recall_loop, f1_loop = indices_confusion_matrix(
        y_true,
        y_pred_loop,
        method_name_loop,
        execution_time_loop,
        df_original_tags,
        output_dir,
        loop_params=loop_params
    )
    return accuracy_loop, precision_loop, recall_loop, f1_loop


# --- Parameter Lists for LoOP Evaluation ---
neighbor_values_loop = [5, 10, 30]  # Example k values for LoOP
lambda_param_values_loop = [2, 3, 4]  # Example lambda_param values for LoOP
loop_results = [] # List to store LoOP results

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


# --- Evaluate LoOP for different parameter values ---
X_loop = df_merged[['X', 'Y', 'Z']].values #df_merged[['X', 'Y', 'Z', 'Red', 'Green', 'Blue']].values if colors is not None else df_merged[['X', 'Y', 'Z']].values # Use RGB if available
scaler = StandardScaler()
X_loop_scaled = scaler.fit_transform(X_loop)
for k in neighbor_values_loop:
    for lambda_param in lambda_param_values_loop:
        accuracy_loop, precision_loop, recall_loop, f1_loop = evaluate_loop_outlier_detection(
            X_loop,
            y_true_original,
            df_merged['Tag'],
            k,
            lambda_param,
            output_dir_loop
        )
        loop_results.append({
            'k': k,
            'lambda_param': lambda_param,
            'accuracy': accuracy_loop,
            'precision': precision_loop,
            'recall': recall_loop,
            'f1_score': f1_loop
        })

print(f"\nResults for LoOP Outlier Detection saved in '{output_dir_loop}' folder.")

# --- Create and Print LoOP Comparison Table ---
loop_results_df = pd.DataFrame(loop_results)
print("\n--- LoOP Parameter Comparison Table ---")
print(loop_results_df)

# --- Find Best LoOP Result ---
best_loop_f1 = 0
best_loop_params = None
for result in loop_results:
    if result['f1_score'] > best_loop_f1:
        best_loop_f1 = result['f1_score']
        best_loop_params = {'k': result['k'], 'lambda_param': result['lambda_param']}

print(f"\n--- Best LoOP Result ---")
print(f"Best parameters: k={best_loop_params['k']}, lambda_param={best_loop_params['lambda_param']} with F1 Score: {best_loop_f1:.4f}")

# --- Plotly Bar Plot for LoOP F1 Scores ---
fig_loop_f1 = px.bar(
    loop_results_df,
    x='k',
    y='f1_score',
    color='lambda_param',
    barmode='group',
    title='LoOP F1 Score vs. k and Lambda',
    labels={'k': 'Number of Neighbors (k)', 'f1_score': 'F1 Score', 'lambda_param': 'Lambda Parameter'}
)
fig_loop_f1.update_layout(yaxis_range=[0, 1])
plot_filename_loop_f1 = os.path.join(output_dir_loop, "LoOP_F1_Score_Comparison_noise_45.png")
fig_loop_f1.write_image(plot_filename_loop_f1)
print(f"LoOP F1 Score bar plot saved in '{output_dir_loop}' folder as '{plot_filename_loop_f1}'.")