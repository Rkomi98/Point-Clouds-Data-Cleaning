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

# --- Output Directories ---
output_dir_iqr = "IQR_Results_Modified_2_Z"  # Changed output directory for modified IQR
output_dir_zscore = "ZScore_Threshold_Comparison_2_Z" # Changed output directory for Z-score threshold comparison
os.makedirs(output_dir_iqr, exist_ok=True)
os.makedirs(output_dir_zscore, exist_ok=True)

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
y_true = np.where(df_merged['Tag'] == 'Outlier', 1, 0)

# --- Outlier Detection Functions (Modified IQR and Z-Score) ---
def detect_outliers_iqr_modified(df, columns):
    """
    Detect outliers using IQR, considering a point as outlier ONLY if it's an outlier
    in ALL specified columns.
    MODIFIED: Only considers 'X' and 'Y' columns.
    """
    columns_to_check = ['Z'] # MODIFICATION:  Focus only on X and Y
    outlier_indices_sets = [] # List to hold outlier indices for each column

    for col in columns_to_check: # MODIFICATION: Iterate over columns_to_check instead of input columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_indices = set(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index) # Use set for intersection later
        outlier_indices_sets.append(outliers_indices)
        print(f"Outliers in '{col}' using IQR method: {len(outliers_indices)} for column '{col}'")

    # Find intersection of outlier indices across all columns
    common_outlier_indices = list(set.intersection(*outlier_indices_sets)) if outlier_indices_sets else [] # Handle empty list case
    print(f"Outliers (in ALL checked columns: X, Y) using Modified IQR method: {len(common_outlier_indices)}") # MODIFICATION: Clarify checked columns in print
    return {"all_columns": common_outlier_indices} # Return as a dictionary for consistency

def detect_outliers_zscore_modified_columns(df, columns, threshold=3): # MODIFICATION: Renamed function to distinguish it
    """
    Detect outliers in the specified DataFrame using the Z-score method,
    but ONLY focusing on 'Blue' and 'Z' columns for outlier detection.

    Parameters:
    - df: The DataFrame containing the data.
    - columns: List of columns (IGNORED in this modified version, kept for signature compatibility).
    - threshold: Z-score threshold beyond which a point is considered an outlier (default is 3).

    Returns:
    - outliers_dict: A dictionary where keys are column names ('Blue', 'Z') and values are indices of outliers
                     detected in each of these columns.
    """
    outliers_dict = {}
    columns_to_check = columns # MODIFICATION: Hardcoded columns to check

    for col in columns_to_check: # MODIFICATION: Iterate over hardcoded columns_to_check
        if col not in df.columns: # MODIFICATION: Check if column exists in DataFrame to avoid errors
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping Z-score for this column.")
            continue

        z_scores = np.abs(stats.zscore(df[col]))
        outliers_indices = np.where(z_scores > threshold)[0]
        outliers_dict[col] = outliers_indices.tolist()
        print(f"Outliers in '{col}' (Blue and Z columns focused Z-Score, threshold={threshold}): {len(outliers_indices)} for column '{col}'") # MODIFICATION: Updated print statement

    return outliers_dict

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None): # Added threshold parameter
    label_mapping = {0: 'Not Outlier', 1: 'Outlier'}
    y_true_str = [label_mapping[label] for label in y_true]
    y_pred_str = [label_mapping[label] for label in y_pred]

    conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=['Not Outlier', 'Outlier'])
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

    threshold_str = f"(threshold={threshold})" if threshold is not None else "" # Add threshold info to output
    print(f"\n--- Metrics for {method_name} {threshold_str}---")
    print("\nMisclassified Tag Distribution:")
    print(tag_counts)

    metrics_filename = os.path.join(output_dir, f"{method_name}{threshold_str}_metrics_2.txt")
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {threshold_str}\n")
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
                xticklabels=['Not Outlier', 'Outlier'], yticklabels=['Not Outlier', 'Outlier'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    title_str = f"Confusion Matrix: {method_name} {threshold_str}" # Add threshold to title
    plt.title(title_str)

    plot_filename = os.path.join(output_dir, f"{method_name}{threshold_str}_confusion_matrix_2.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1 # Return metrics for comparison

def plot_outliers_boxplot(df, columns_to_check, iqr_outliers, zscore_outliers, output_dir_boxplot):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_check):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue", fliersize=0)
        axes[i].set_title(f"Boxplot of {col}")
        y_position = axes[i].get_ylim()[1] * 0.01

        if col in iqr_outliers and iqr_outliers[col]: # Check if column exists and has outliers
            iqr_outlier_values = df.loc[iqr_outliers[col], col]
            axes[i].plot(iqr_outlier_values, [y_position] * len(iqr_outlier_values), 'ro', markersize=5, label="IQR Outliers")

        if col in zscore_outliers and zscore_outliers[col]: # Check if column exists and has outliers
            zscore_outlier_values = df.loc[zscore_outliers[col], col]
            axes[i].plot(zscore_outlier_values, [y_position] * len(zscore_outlier_values), 'go', markersize=5, label="Z-score Outliers")

        axes[i].legend()

    plt.tight_layout()
    boxplot_filename = os.path.join(output_dir_boxplot, "outlier_boxplots_2.png")
    plt.savefig(boxplot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_distribution(df, columns_to_check, output_dir):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_check):
        sns.violinplot(x=df[col], ax=axes[i], color="lightcoral")
        axes[i].set_title(f"Violin Plot of {col}")

    plt.tight_layout()
    violinplot_filename = os.path.join(output_dir, "data_distribution_violinplots_2.png")
    plt.savefig(violinplot_filename, dpi=300, bbox_inches='tight')
    plt.close()

# --- Columns for Outlier Detection ---
columns_to_check = ['X', 'Y','Z']
#columns_to_check = ['X', 'Y', 'Z', 'Red', 'Green', 'Blue']

# --- Plot Data Distribution (Violin Plots) ---
plot_data_distribution(df_merged, columns_to_check, output_dir_iqr) # Violin plot saved in IQR dir (can be any result dir)

# --- Modified IQR Outlier Detection and Evaluation ---
start_time_iqr = time.time()
iqr_outliers_indices_dict_modified = detect_outliers_iqr_modified(df_merged, columns=['Z'])

# Create y_pred for Modified IQR
predicted_labels_iqr_modified = np.zeros(len(df_merged), dtype=int)
predicted_labels_iqr_modified[iqr_outliers_indices_dict_modified['all_columns']] = 1 # Use 'all_columns' key

end_time_iqr = time.time()
execution_time_iqr = end_time_iqr - start_time_iqr
method_name_iqr = 'Modified IQR Method'

plot_outliers_boxplot(df_merged, columns_to_check, iqr_outliers_indices_dict_modified, {}, output_dir_iqr) # Boxplot for modified IQR

accuracy_iqr_mod, precision_iqr_mod, recall_iqr_mod, f1_iqr_mod = indices_confusion_matrix(y_true, predicted_labels_iqr_modified, method_name_iqr, execution_time_iqr, df_merged['Tag'], output_dir_iqr)

# --- Z-Score Outlier Detection and Evaluation with Threshold Comparison ---
zscore_thresholds = [3,3.2,3.4,3.6,3.8,4] # Different thresholds to test
zscore_results = [] # List to store results for each threshold

for threshold in zscore_thresholds:
    start_time_zscore = time.time()
    zscore_outliers_indices_dict = detect_outliers_zscore_modified_columns(df_merged, columns_to_check, threshold=threshold)

    # Create y_pred for Z-Score
    predicted_labels_zscore = np.zeros(len(df_merged), dtype=int)
    for col in zscore_outliers_indices_dict:
        predicted_labels_zscore[zscore_outliers_indices_dict[col]] = 1

    end_time_zscore = time.time()
    execution_time_zscore = end_time_zscore - start_time_zscore
    method_name_zscore = 'Z-Score Method'

    plot_outliers_boxplot(df_merged, columns_to_check, {}, zscore_outliers_indices_dict, output_dir_zscore) # Boxplot for Z-score

    accuracy_zscore, precision_zscore, recall_zscore, f1_zscore = indices_confusion_matrix(y_true, predicted_labels_zscore, method_name_zscore, execution_time_zscore, df_merged['Tag'], output_dir_zscore, threshold=threshold)
    zscore_results.append({ # Store results
        'threshold': threshold,
        'accuracy': accuracy_zscore,
        'precision': precision_zscore,
        'recall': recall_zscore,
        'f1_score': f1_zscore,
        'execution_time': execution_time_zscore
    })

print(f"\nResults for Modified IQR saved in '{output_dir_iqr}' folder.")
print(f"Results for Z-Score Threshold Comparison saved in '{output_dir_zscore}' folder.")

# --- Print Z-Score Comparison Summary ---
print("\n--- Z-Score Threshold Comparison Summary ---")
for result in zscore_results:
    print(f"Threshold: {result['threshold']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")
    print(f"  Execution Time: {result['execution_time']:.4f} seconds")
    print("-" * 30)

print("\n--- Modified IQR Method Summary ---")
print(f"  Accuracy: {accuracy_iqr_mod:.4f}")
print(f"  Precision: {precision_iqr_mod:.4f}")
print(f"  Recall: {recall_iqr_mod:.4f}")
print(f"  F1 Score: {f1_iqr_mod:.4f}")
print(f"  Execution Time: {execution_time_iqr:.4f} seconds")