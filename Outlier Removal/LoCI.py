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
from pyod.models.loci import LOCI # Import LoCI from PyOD
import plotly.express as px

# --- Output Directories ---
output_dir_loci = "LoCI_Results_45_Downsampled_Debug" # Changed output directory for debugging
os.makedirs(output_dir_loci, exist_ok=True)

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

# --- DEBUG: Limit dataset size for testing ---
DEBUG_DATA_LIMIT = True # Set to True to limit data for faster debugging - ENABLED FOR DEBUG
DEBUG_DATA_POINT_COUNT = 5000 # Reduced to 5000 points for faster testing
if DEBUG_DATA_LIMIT:
    df_merged = df_merged.iloc[:DEBUG_DATA_POINT_COUNT] # Use iloc for integer-based indexing
    print(f"DEBUG MODE: Using only the first {len(df_merged)} points for testing!")


# Corrected line: Use parentheses for each condition and then bitwise OR
y_true_original = np.where((df_merged['Tag'] == 'Outlier') | (df_merged['Tag'] == 'Noise'), 1, 0)

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None, loci_params=None): # Changed lof_params to loci_params
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
    loci_param_str = f"(k={loci_params['k']}, alpha={loci_params['alpha']}, downsample_voxel_size={loci_params.get('downsample_voxel_size', 'None')})" if loci_params else "" # Added downsample info
    params_str_for_filename = f"{threshold_str}{loci_param_str}".replace("(", "").replace(")", "").replace(",", "_").replace("=", "_").replace(".", "p") # Create filename string


    print(f"\n--- Metrics for {method_name} {params_str_for_filename} ---") # Changed method_name to LoCI
    print("\nMisclassified Tag Distribution:")
    for tag, count in tag_counts.items():
        percentage = misclassified_tag_percentages[tag]
        print(f"  {tag}: {count} points ({percentage:.2f}%)")

    metrics_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_metrics_noise_45.txt") # Changed method_name to LoCI
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {params_str_for_filename} (Detecting Outlier/Noise)\n") # Changed method_name to LoCI
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
    title_str = f"Confusion Matrix: {method_name} {params_str_for_filename} (Outlier/Noise Detection)" # Changed method_name to LoCI
    plt.title(title_str)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plot_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_confusion_matrix_noise_45.png") # Changed method_name to LoCI
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1

# --- LoCI Outlier Detection Evaluation Function ---
def evaluate_loci_outlier_detection(X_original, y_true, df_original_tags, k, alpha, output_dir, downsample_voxel_size=None): # Added downsample_voxel_size
    """
    Evaluates Local Correlation Integral (LoCI) for outlier detection with optional downsampling.
    """
    start_time_loci_function = time.time() # START TIMER FUNCTION

    print(f"  Evaluating LoCI with k={k}, alpha={alpha}, voxel_size={downsample_voxel_size}...") # DEBUG PRINT START

    if downsample_voxel_size is not None:
        pcd_downsampled = o3d.geometry.PointCloud()
        pcd_downsampled.points = o3d.utility.Vector3dVector(X_original[:, :3]) # Use only XYZ for downsampling
        print("    Starting Voxel Downsampling...") # DEBUG PRINT
        pcd_downsampled = pcd_downsampled.voxel_down_sample(voxel_size=downsample_voxel_size)
        X_downsampled = np.asarray(pcd_downsampled.points)
        if X_original.shape[1] > 3: # If colors are present, downsample colors as well (crude approximation)
            colors_original = X_original[:, 3:]
            num_original_points = X_original.shape[0]
            num_downsampled_points = X_downsampled.shape[0]
            color_indices = np.linspace(0, num_original_points-1, num_downsampled_points, dtype=int) # Simple linear interpolation of color indices
            colors_downsampled = colors_original[color_indices]
            X = np.concatenate((X_downsampled, colors_downsampled), axis=1)
        else:
            X = X_downsampled
        print(f"    Point cloud downsampled to {X.shape[0]} points using voxel size: {downsample_voxel_size}") # DEBUG PRINT
    else:
        X = X_original # Use original data if no downsampling

    print("    Initializing LoCI detector...") # DEBUG PRINT
    loci_detector = LOCI(k=k, alpha=alpha) # Initialize LoCI detector with correct parameter k
    print("    Fitting LoCI...") # DEBUG PRINT
    loci_detector.fit(X) # Fit LoCI
    print("    Predicting outliers with LoCI...") # DEBUG PRINT
    y_pred_loci = loci_detector.predict(X) # Predict outliers (0: inliers, 1: outliers)

    # Need to map predictions back to the original point cloud if downsampling was used.
    if downsample_voxel_size is not None:
        y_pred_loci_original_scale = np.zeros(len(y_true), dtype=int) # Assume all inliers initially
        # This is a SIMPLIFIED mapping - assumes downsampling roughly preserves index order. More robust mapping needed for precise evaluation.
        y_pred_loci_original_scale[np.linspace(0, len(y_true)-1, len(y_pred_loci), dtype=int)] = y_pred_loci
        y_pred_loci_final = y_pred_loci_original_scale
    else:
        y_pred_loci_final = y_pred_loci


    end_time_loci_function = time.time() # END TIMER FUNCTION
    execution_time_loci = end_time_loci_function - start_time_loci_function
    method_name_loci = 'LoCI Outlier Detection'

    loci_params = {'k': k, 'alpha': alpha, 'downsample_voxel_size': downsample_voxel_size} # Include downsample_voxel_size in params
    print("    Evaluating metrics...") # DEBUG PRINT
    accuracy_loci, precision_loci, recall_loci, f1_loci = indices_confusion_matrix(
        y_true,
        y_pred_loci_final, # Use mapped predictions for evaluation
        method_name_loci,
        execution_time_loci,
        df_original_tags,
        output_dir,
        loci_params=loci_params # Pass loci_params
    )
    print(f"  LoCI evaluation completed in {execution_time_loci:.2f} seconds.") # DEBUG PRINT END
    return accuracy_loci, precision_loci, recall_loci, f1_loci


# --- Parameter Lists for LoCI Evaluation ---
neighbor_values_loci = [5, 10, 15]  # Example k values for LoCI (renamed for clarity) - REDUCED to very small values
alpha_values_loci = [0.7, 0.9] # Example alpha values for LoCI - Reduced range
downsample_voxel_sizes = [0.8, 1.2, None] # Example voxel sizes, None for no downsampling - REDUCED RANGE & ADJUSTED VALUES
loci_results = [] # List to store LoCI results

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


# --- Evaluate LoCI for different parameter values ---
X_loci_original = df_merged[['X', 'Y', 'Z']].values # Use RGB if available df_merged[['X', 'Y', 'Z', 'Red', 'Green', 'Blue']].values if colors is not None else 

start_time_total = time.time() # START TIMER TOTAL

for k in neighbor_values_loci: # Changed n_neighbors to k
    for alpha in alpha_values_loci:
        for downsample_voxel_size in downsample_voxel_sizes:
            print(f"\nStarting LoCI evaluation for k={k}, alpha={alpha}, voxel_size={downsample_voxel_size}...") # DEBUG PRINT OUTER LOOP START
            accuracy_loci, precision_loci, recall_loci, f1_loci = evaluate_loci_outlier_detection(
                X_loci_original, # Pass original data for downsampling inside function
                y_true_original,
                df_merged['Tag'],
                k, # Changed n_neighbors to k
                alpha,
                output_dir_loci,
                downsample_voxel_size=downsample_voxel_size # Pass voxel size
            )
            loci_results.append({
                'k': k, # Changed n_neighbors to k in results dict
                'alpha': alpha,
                'downsample_voxel_size': downsample_voxel_size,
                'accuracy': accuracy_loci,
                'precision': precision_loci,
                'recall': recall_loci,
                'f1_score': f1_loci
            })
            print(f"Finished LoCI evaluation for k={k}, alpha={alpha}, voxel_size={downsample_voxel_size}.\n") # DEBUG PRINT OUTER LOOP END

end_time_total = time.time() # END TIMER TOTAL
execution_time_total = end_time_total - start_time_total
print(f"\nTotal LoCI evaluation time: {execution_time_total:.2f} seconds.") # DEBUG PRINT TOTAL TIME


print(f"\nResults for LoCI Outlier Detection (Downsampled) saved in '{output_dir_loci}' folder.")


# --- Print LoCI Comparison Summary and Find Best F1 Score ---
print("\n--- LoCI Parameter Comparison Summary (Downsampled) ---")
best_loci_f1 = 0
best_loci_params = None
for result in loci_results:
    print(f"k: {result['k']}, alpha: {result['alpha']}, voxel_size: {result['downsample_voxel_size']}") # Added voxel_size to print
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print("-" * 40)
    if result['f1_score'] > best_loci_f1:
        best_loci_f1 = result['f1_score']
        best_loci_params = {'k': result['k'], 'alpha': result['alpha'], 'downsample_voxel_size': result['downsample_voxel_size']} # Added voxel_size to best params

print(f"\n--- Best LoCI Result (Downsampled) ---")
print(f"Best parameters: k={best_loci_params['k']}, alpha={best_loci_params['alpha']}, voxel_size={best_loci_params['downsample_voxel_size']} with F1 Score: {best_loci_f1:.4f}") # Added voxel_size to best result print

# --- Plotly Bar Plot for LoCI F1 Scores (Example - Adjust as needed for alpha and voxel_size) ---
loci_f1_df = pd.DataFrame(loci_results)
fig_loci_f1 = px.bar(
    loci_f1_df,
    x='k', # Changed n_neighbors to k in plot
    y='f1_score',
    color='alpha', # Color bars by alpha value
    facet_col='downsample_voxel_size', # Facet by voxel size
    barmode='group', # Group bars for different alpha values side-by-side
    title='LoCI F1 Score vs. k, Alpha, and Downsample Voxel Size', # Changed title to include voxel_size
    labels={'k': 'Number of Neighbors (k)', 'f1_score': 'F1 Score', 'alpha': 'Alpha', 'downsample_voxel_size': 'Voxel Size'} # Changed label to k and added voxel_size
)
fig_loci_f1.update_layout(yaxis_range=[0, 1]) # Ensure y-axis range is 0 to 1 for F1 score
plot_filename_loci_f1 = os.path.join(output_dir_loci, "LoCI_F1_Score_Comparison_downsampled_noise_45.png") # Changed filename
fig_loci_f1.write_image(plot_filename_loci_f1)
print(f"LoCI F1 Score bar plot (Downsampled) saved in '{output_dir_loci}' folder as '{plot_filename_loci_f1}'.") # Changed print message