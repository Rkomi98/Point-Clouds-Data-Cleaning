import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import os
import plotly.express as px

# --- Output Directories ---
output_dir_covariance_feature = "CovarianceFeature_Results_2" # New output directory for combined Covariance Feature method
os.makedirs(output_dir_covariance_feature, exist_ok=True)

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

def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags, output_dir, threshold=None, covariance_params=None): # MODIFICATION: Added covariance_params
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
    covariance_param_str = f"(k={covariance_params['k']}, feature_used={covariance_params.get('feature_used', 'None')}, threshold_percentile={covariance_params.get('threshold_percentile', 'None')})" if covariance_params else "" # MODIFICATION: Added feature_used to params string
    params_str_for_filename = f"{threshold_str}{covariance_param_str}".replace("(", "").replace(")", "").replace(",", "_").replace("=", "_").replace(".", "p") # Create filename string

    print(f"\n--- Metrics for {method_name} {params_str_for_filename} ---") # MODIFICATION: Changed method_name to Covariance
    print("\nMisclassified Tag Distribution:")
    for tag, count in tag_counts.items():
        percentage = misclassified_tag_percentages[tag]
        print(f"  {tag}: {count} points ({percentage:.2f}%)")

    metrics_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_metrics_noise_45.txt") # MODIFICATION: Changed method_name to Covariance and filename to include covariance_params
    with open(metrics_filename, "w") as f:
        f.write(f"Metrics for {method_name} {params_str_for_filename} (Detecting Outlier/Noise)\n") # MODIFICATION: Changed method_name to Covariance
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
    title_str = f"Confusion Matrix: {method_name} {params_str_for_filename} (Outlier/Noise Detection)" # MODIFICATION: Changed method_name to Covariance
    plt.title(title_str)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plot_filename = os.path.join(output_dir, f"{method_name}_{params_str_for_filename}_confusion_matrix_noise_45.png") # MODIFICATION: Changed method_name to Covariance and filename to include covariance_params
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return accuracy, precision, recall, f1

def extract_covariance_features(pcd_data, k):
    """
    Extract covariance features (eigenvalues and dimensionality features) for each point.
    """
    points = np.asarray(pcd_data.points)
    num_points = len(points)
    features = np.zeros((num_points, 8))  # Store eigenvalues (e1, e2, e3) and dimensionality features (L, P, A, O, E)

    kdtree = o3d.geometry.KDTreeFlann(pcd_data)

    for i in range(num_points):
        _, knn_indices, _ = kdtree.search_knn_vector_3d(pcd_data.points[i], k)
        knn_points = points[knn_indices[1:]] # Exclude the point itself

        if len(knn_points) < 3: # Need at least 3 points to compute covariance
            features[i] = np.array([0] * 8) # Or handle appropriately, setting all features to 0
            continue

        cov_matrix = np.cov(knn_points.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1] # Sort eigenvalues in descending order (e1 >= e2 >= e3)
        eigenvalues = np.clip(eigenvalues, 0, None) # Ensure eigenvalues are non-negative

        # --- Dimensionality Features Calculation (Table 1 from Paper) ---
        l1, l2, l3 = eigenvalues # Assign eigenvalues to l1, l2, l3 for clarity
        mu = l1 # Use largest eigenvalue as mu (as in paper's equations, though 'mu' is somewhat ambiguously defined there - needs clarification if different in full paper)

        linearity = (l1 - l2) / mu if mu > 0 else 0
        planarity = (l2 - l3) / mu if mu > 0 else 0
        anisotropy = (l1 - l3) / mu if mu > 0 else 0
        omnivariance = (l1 * l2 * l3)**(1/3)
        eigenentropy = -np.sum([li * np.log(li) if li > 0 else 0 for li in eigenvalues]) # Avoid log(0)

        features[i] = np.array([eigenvalues[0], eigenvalues[1], eigenvalues[2], linearity, planarity, anisotropy, omnivariance, eigenentropy])

    return features

def evaluate_covariance_outlier_detection(df_merged, y_true, df_original_tags, k_values, output_dir, threshold_percentiles, feature_choices): # MODIFICATION: Added feature_choices
    """
    Evaluates covariance feature based outlier detection using Eigenentropy or Omnivariance for scoring.
    """
    f1_scores = []
    pcd_data = o3d.geometry.PointCloud()
    pcd_data.points = o3d.utility.Vector3dVector(df_merged[['X', 'Y', 'Z']].values)

    for k in k_values:
        features = extract_covariance_features(pcd_data, k)
        for feature_choice in feature_choices: # MODIFICATION: Loop through feature_choices
            if feature_choice == 'Eigenentropy':
                feature_values = features[:, 7] # Eigenentropy is the 7th feature (index 6)
                method_name_covariance = 'Covariance Outlier Detection (Eigenentropy)'
            elif feature_choice == 'Omnivariance':
                feature_values = features[:, 6] # Omnivariance is the 6th feature (index 5) - CORRECTED INDEX
                method_name_covariance = 'Covariance Outlier Detection (Omnivariance)' # MODIFICATION: Updated method name

            for threshold_percentile in threshold_percentiles: # Loop through different threshold percentiles
                start_time_covariance = time.time()


                # --- Thresholding based on Feature percentile ---
                threshold_value = np.percentile(feature_values, threshold_percentile) # Threshold based on percentile
                y_pred_covariance = np.where(feature_values > threshold_value, 1, 0) # 1 for outlier

                end_time_covariance = time.time()
                execution_time_covariance = end_time_covariance - start_time_covariance

                covariance_params = {'k': k, 'threshold_percentile': threshold_percentile, 'feature_used': feature_choice} # MODIFICATION: Include feature_used in params
                accuracy_covariance, precision_covariance, recall_covariance, f1_covariance = indices_confusion_matrix(
                    y_true,
                    y_pred_covariance,
                    method_name_covariance,
                    execution_time_covariance,
                    df_original_tags,
                    output_dir,
                    covariance_params=covariance_params
                )
                f1_scores.append({ # Append dictionary for better table
                    'k': k,
                    'threshold_percentile': threshold_percentile, # Store threshold_percentile in results
                    'feature_used': feature_choice, # MODIFICATION: Store feature_used in results
                    'accuracy': accuracy_covariance,
                    'precision': precision_covariance,
                    'recall': recall_covariance,
                    'f1_score': f1_covariance
                })
    return f1_scores

# --- Parameter Lists for Covariance Evaluation ---
k_values_covariance = [5, 10, 20, 30] # Example k values for covariance feature extraction - Reduced range
threshold_percentiles = [75, 90, 95, 98] # Example percentile thresholds for Eigenentropy
feature_choices = ['Eigenentropy', 'Omnivariance'] # MODIFICATION: List of feature choices to evaluate
covariance_results_f1 = []

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


# --- Evaluate Covariance for different parameter values ---
covariance_results_f1 = evaluate_covariance_outlier_detection(df_merged, y_true_original, df_merged['Tag'], k_values_covariance, output_dir_covariance_feature, threshold_percentiles, feature_choices) # MODIFICATION: Pass feature_choices

print(f"\nResults for Covariance Outlier Detection (Eigenentropy and Omnivariance) saved in '{output_dir_covariance_feature}' folder.") # MODIFICATION: Updated output dir name in print

# --- Create and Print Covariance Comparison Table ---
covariance_results_df = pd.DataFrame(covariance_results_f1)
print("\n--- Covariance Parameter Comparison Table (Eigenentropy and Omnivariance) ---") # MODIFICATION: Updated table name in print
print(covariance_results_df)

# --- Find Best Covariance Result ---
best_covariance_f1 = 0
best_covariance_params = None
for result in covariance_results_f1:
    if result['f1_score'] > best_covariance_f1:
        best_covariance_f1 = result['f1_score']
        best_covariance_params = {'k': result['k'], 'threshold_percentile': result['threshold_percentile'], 'feature_used': result['feature_used']} # MODIFICATION: Include feature_used in best params

print(f"\n--- Best Covariance Result (Eigenentropy and Omnivariance) ---") # MODIFICATION: Updated result name in print
print(f"Best parameters: k={best_covariance_params['k']}, threshold_percentile={best_covariance_params['threshold_percentile']}, feature_used={best_covariance_params['feature_used']} with F1 Score: {best_covariance_f1:.4f}") # MODIFICATION: Include feature_used in best result print

# --- Plotly Bar Plot for Covariance F1 Scores ---
fig_covariance_f1 = px.bar(
    covariance_results_df,
    x='k',
    y='f1_score',
    color='threshold_percentile', # MODIFICATION: Color by threshold_percentile
    facet_col='feature_used', # MODIFICATION: Facet by feature_used
    barmode='group',
    title='Covariance F1 Score vs. k, Threshold Percentile, and Feature', # MODIFICATION: Updated title
    labels={'k': 'Neighbors (k)', 'f1_score': 'F1 Score', 'threshold_percentile': 'Threshold Percentile', 'feature_used': 'Feature'} # MODIFICATION: Updated labels
)
fig_covariance_f1.update_layout(yaxis_range=[0, 1])
plot_filename_covariance_f1 = os.path.join(output_dir_covariance_feature, "Covariance_F1_Score_Comparison_Feature_noise_45.png") # MODIFICATION: Updated filename
fig_covariance_f1.write_image(plot_filename_covariance_f1)
print(f"Covariance F1 Score bar plot (Eigenentropy and Omnivariance) saved in '{output_dir_covariance_feature}' folder as '{plot_filename_covariance_f1}'.") # MODIFICATION: Updated print message