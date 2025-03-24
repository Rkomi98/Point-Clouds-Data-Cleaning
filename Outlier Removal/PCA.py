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

# --- Output Directories ---
output_dir_pca = "PCA_Outlier_Removal_2"
os.makedirs(output_dir_pca, exist_ok=True)

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
y_true = np.where((df_merged['Tag'] == 'Outlier') | (df_merged['Tag'] == 'Noise'), 1, 0)

# --- Outlier Detection Functions ---
def detect_outliers_iqr_modified(df, columns):
    columns_to_check = ['X', 'Y']
    outlier_indices_sets = []
    for col in columns_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_indices = set(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)
        outlier_indices_sets.append(outliers_indices)
        print(f"Outliers in '{col}' using IQR method: {len(outliers_indices)} for column '{col}'")

    common_outlier_indices = list(set.intersection(*outlier_indices_sets)) if outlier_indices_sets else []
    print(f"Outliers (in ALL checked columns: X, Y) using Modified IQR method: {len(common_outlier_indices)}")
    return {"all_columns": common_outlier_indices}

def detect_outliers_zscore_modified_columns(df, columns, threshold=3):
    outliers_dict = {}
    columns_to_check = ['Blue', 'Z']

    for col in columns_to_check:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping Z-score for this column.")
            continue

        z_scores = np.abs(stats.zscore(df[col]))
        outliers_indices = np.where(z_scores > threshold)[0]
        outliers_dict[col] = outliers_indices.tolist()
        print(f"Outliers in '{col}' (Blue and Z columns focused Z-Score, threshold={threshold}): {len(outliers_indices)} for column '{col}'")

    return outliers_dict

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None):
    label_mapping = {0: 'Not Outlier/Noise', 1: 'Outlier/Noise'}
    y_true_str = [label_mapping[label] for label in y_true]
    y_pred_str = [label_mapping[label] for label in y_pred]

    conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=['Not Outlier/Noise', 'Outlier/Noise'])
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    accuracy = accuracy_score(y_true_str, y_pred_str)
    precision = precision_score(y_true_str, y_pred_str, pos_label='Outlier/Noise')
    recall = recall_score(y_true_str, y_pred_str, pos_label='Outlier/Noise')
    f1 = f1_score(y_true_str, y_pred_str, pos_label='Outlier/Noise')

    misclassified_mask = (y_true != y_pred)
    misclassified_tags = original_tags[misclassified_mask]
    tag_counts = pd.Series(misclassified_tags).value_counts()

    threshold_str = f"(threshold={threshold})" if threshold is not None else ""
    print(f"\n--- Metrics for {method_name} {threshold_str}---")
    print("\nMisclassified Tag Distribution:")
    print(tag_counts)

    metrics_filename = os.path.join(output_dir, f"{method_name}{threshold_str}_metrics_noise_2.txt")
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {threshold_str} (Detecting Outlier/Noise)\n")
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
    title_str = f"Confusion Matrix: {method_name} {threshold_str} (Outlier/Noise Detection)"
    plt.title(title_str)

    plot_filename = os.path.join(output_dir, f"{method_name}{threshold_str}_confusion_matrix_noise_2.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1

def plot_outliers_boxplot(df, columns_to_check, iqr_outliers, zscore_outliers, output_dir_boxplot):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_check):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue", fliersize=0)
        axes[i].set_title(f"Boxplot of {col}")
        y_position = axes[i].get_ylim()[1] * 0.01

        if col in iqr_outliers and iqr_outliers[col]:
            iqr_outlier_values = df.loc[iqr_outliers[col], col]
            axes[i].plot(iqr_outlier_values, [y_position] * len(iqr_outlier_values), 'ro', markersize=5, label="IQR Outliers")

        if col in zscore_outliers and zscore_outliers[col]:
            zscore_outlier_values = df.loc[zscore_outliers[col], col]
            axes[i].plot(zscore_outlier_values, [y_position] * len(zscore_outlier_values), 'go', markersize=5, label="Z-score Outliers")

        axes[i].legend()

    plt.tight_layout()
    boxplot_filename = os.path.join(output_dir_boxplot, "outlier_boxplots_noise_2.png")
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_distribution(df, columns_to_check, output_dir):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_check):
        sns.violinplot(x=df[col], ax=axes[i], color="lightcoral")
        axes[i].set_title(f"Violin Plot of {col}")

    plt.tight_layout()
    violinplot_filename = os.path.join(output_dir, "data_distribution_violinplots_noise_2.png")
    plt.savefig(violinplot_filename, dpi=300, bbox_inches='tight')
    plt.close()

# --- PCA Outlier Removal Function ---
def remove_outliers_pca(df, columns_pca=['X', 'Y', 'Z'], n_components=3, error_threshold_factor=1.5):
    """
    Removes outliers based on PCA reconstruction error using IQR threshold.

    Parameters:
    - df: DataFrame
    - columns_pca: Columns to use for PCA.
    - n_components: Number of principal components to keep for reconstruction.
    - error_threshold_factor: Factor for IQR to determine error threshold.

    Returns:
    - outlier_indices: List of indices identified as outliers.
    """
    data_pca = df[columns_pca].copy()
    if data_pca.isnull().values.any(): # Handle NaN values if any exist
        data_pca.fillna(data_pca.mean(), inplace=True) # Simple imputation, consider more sophisticated methods

    scaler = StandardScaler() # Standardize the data before PCA
    data_scaled = scaler.fit_transform(data_pca)

    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_scaled)
    data_reconstructed = pca.inverse_transform(data_reduced)
    data_reconstructed_original_scale = scaler.inverse_transform(data_reconstructed) # Inverse transform back to original scale

    reconstruction_errors = np.sum(np.square(data_scaled - data_reconstructed), axis=1) # Error calculation on scaled data

    Q1 = pd.Series(reconstruction_errors).quantile(0.25)
    Q3 = pd.Series(reconstruction_errors).quantile(0.75)
    IQR = Q3 - Q1
    error_threshold = Q3 + error_threshold_factor * IQR

    outlier_indices = np.where(reconstruction_errors > error_threshold)[0]
    print(f"PCA Outlier Detection - Number of outliers removed: {len(outlier_indices)}")
    return outlier_indices.tolist()


# --- Columns for Outlier Detection ---
columns_to_check = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue']
columns_pca = ['X', 'Y', 'Z'] # Columns used for PCA

# --- Plot Data Distribution (Violin Plots) ---
plot_data_distribution(df_merged, columns_to_check, output_dir_pca)

# --- PCA Outlier Removal and Evaluation ---
start_time_pca = time.time()
pca_outlier_indices = remove_outliers_pca(df_merged, columns_pca=columns_pca, n_components=3, error_threshold_factor=1.5) # Using error_threshold_factor

# Create y_pred for PCA
predicted_labels_pca = np.zeros(len(df_merged), dtype=int)
predicted_labels_pca[pca_outlier_indices] = 1

end_time_pca = time.time()
execution_time_pca = end_time_pca - start_time_pca
method_name_pca = 'PCA Outlier Removal'

# No boxplot for PCA outliers directly as it's based on reconstruction error, not individual columns

accuracy_pca, precision_pca, recall_pca, f1_pca = indices_confusion_matrix(y_true, predicted_labels_pca, method_name_pca, execution_time_pca, df_merged['Tag'], output_dir_pca)


print(f"\nResults for PCA Outlier Removal saved in '{output_dir_pca}' folder.")

print("\n--- PCA Outlier Removal Summary ---")
print(f"  Accuracy: {accuracy_pca:.4f}")
print(f"  Precision: {precision_pca:.4f}")
print(f"  Recall: {recall_pca:.4f}")
print(f"  F1 Score: {f1_pca:.4f}")
print(f"  Execution Time: {execution_time_pca:.4f} seconds")