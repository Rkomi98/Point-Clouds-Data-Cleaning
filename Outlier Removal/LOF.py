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
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px

# --- Output Directories ---
output_dir_lof = "LOF_Results_XYZ_2" # Output directory for LOF
os.makedirs(output_dir_lof, exist_ok=True)

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
# Corrected line: Use parentheses for each condition and then bitwise OR
y_true_original = np.where((df_merged['Tag'] == 'Outlier') | (df_merged['Tag'] == 'Noise'), 1, 0)

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None, lof_params=None): # Removed sor_params
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
    lof_param_str = f"(n_neighbors={lof_params['n_neighbors']})" if lof_params else "" # MODIFICATION: Kept LOF params string
    params_str_for_filename = f"{threshold_str}{lof_param_str}".replace("(", "").replace(")", "").replace(",", "_").replace("=", "_") # Create a string for filename


    print(f"\n--- Metrics for {method_name} {params_str_for_filename} ---")
    print("\nMisclassified Tag Distribution:")
    for tag, count in tag_counts.items():
        percentage = misclassified_tag_percentages[tag]
        print(f"  {tag}: {count} points ({percentage:.2f}%)")

    metrics_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_metrics_noise_45.txt") # MODIFICATION: Removed SOR params from filename
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {params_str_for_filename} (Detecting Outlier/Noise)\n") # MODIFICATION: Removed SOR params from header
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
    title_str = f"Confusion Matrix: {method_name} {params_str_for_filename} (Outlier/Noise Detection)" # MODIFICATION: Removed SOR params from title
    plt.title(title_str)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plot_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_confusion_matrix_noise_2.png") # MODIFICATION: Removed SOR params from filename
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1

# --- LOF Outlier Detection Evaluation Function ---
def evaluate_lof_outlier_detection(X, y_true, df_original_tags, n_neighbors, output_dir):
    """
    Evaluates Local Outlier Factor (LOF) for outlier detection.
    """
    start_time_lof = time.time()
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    y_pred_lof = lof.fit_predict(X)
    y_pred_lof_outlier_label = np.where(y_pred_lof == -1, 1, 0)  # LOF predicts -1 for outliers, we map it to 1
    end_time_lof = time.time()
    execution_time_lof = end_time_lof - start_time_lof
    method_name_lof = 'LOF Outlier Detection'

    lof_params = {'n_neighbors': n_neighbors}
    accuracy_lof, precision_lof, recall_lof, f1_lof = indices_confusion_matrix(
        y_true,
        y_pred_lof_outlier_label,
        method_name_lof,
        execution_time_lof,
        df_original_tags,
        output_dir,
        lof_params=lof_params
    )
    return accuracy_lof, precision_lof, recall_lof, f1_lof


# --- Parameter Lists for LOF Evaluation ---
neighbor_values = [2, 5, 10, 15, 20, 30, 40, 45, 100, 150]  # List of `n_neighbors` values to test
#neighbor_values = [40,45,50,60,70,80,90]  # List of `n_neighbors` values to test
lof_results = [] # List to store LOF results

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


# --- Evaluate LOF for different neighbor values ---
X_lof = df_merged[['X', 'Y', 'Z']].values if colors is not None else df_merged[['X', 'Y', 'Z']].values # Use RGB if available
for n_neighbors in neighbor_values:
    accuracy_lof, precision_lof, recall_lof, f1_lof = evaluate_lof_outlier_detection(
        X_lof,
        y_true_original,
        df_merged['Tag'],
        n_neighbors,
        output_dir_lof
    )
    lof_results.append({
        'n_neighbors': n_neighbors,
        'accuracy': accuracy_lof,
        'precision': precision_lof,
        'recall': recall_lof,
        'f1_score': f1_lof
    })

print(f"\nResults for LOF Outlier Detection saved in '{output_dir_lof}' folder.")


# --- Print LOF Comparison Summary and Find Best F1 Score ---
print("\n--- LOF Parameter Comparison Summary ---")
best_lof_f1 = 0
best_lof_n_neighbors = None
for result in lof_results:
    print(f"n_neighbors: {result['n_neighbors']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print("-" * 40)
    if result['f1_score'] > best_lof_f1:
        best_lof_f1 = result['f1_score']
        best_lof_n_neighbors = result['n_neighbors']

print(f"\n--- Best LOF Result ---")
print(f"Best n_neighbors: {best_lof_n_neighbors} with F1 Score: {best_lof_f1:.4f}")

# --- Plotly Bar Plot for LOF F1 Scores ---
fig_lof_f1 = px.bar(
    lof_results,
    x='n_neighbors',
    y='f1_score',
    title='LOF F1 Score vs. n_neighbors',
    labels={'n_neighbors': 'Number of Neighbors', 'f1_score': 'F1 Score'}
)
fig_lof_f1.update_layout(yaxis_range=[0, 1]) # Ensure y-axis range is 0 to 1 for F1 score
plot_filename_lof_f1 = os.path.join(output_dir_lof, "LOF_F1_Score_Comparison_noise_45.png")
fig_lof_f1.write_image(plot_filename_lof_f1)
print(f"LOF F1 Score bar plot saved in '{output_dir_lof}' folder as '{plot_filename_lof_f1}'.")