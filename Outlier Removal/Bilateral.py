import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import laspy
import pandas as pd
import polars as pl
import numpy as np
import open3d as o3d
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import confusion_matrix
import time  # Import for time measurement

def create_output_folder(folder_path):
    """Creates an output folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def log_to_file(log_file, message):
    """Appends a log message to a log file."""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def import_las_to_dataframe(file_path):
    """
    Imports a LAS file and converts it into a Polars DataFrame with X, Y, Z, R, G, B, and Tag columns.
    """
    # Read the LAS file
    las_file = laspy.read(file_path)

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
        tags_numeric = np.zeros(points.shape[0], dtype=int)
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

    return df

def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    """
    Adds noisy points to a *copy* of a subset of the input DataFrame, tagging them as 'Noise',
    and preserves the original DataFrame's tags for non-noisy points. (CORRECTED TAG HANDLING)
    """
    num_points = df.shape[0]
    num_noisy_points = int(noise_percentage / 100 * num_points)
    indices = np.random.choice(num_points, num_noisy_points, replace=False)

    # Select rows by index using iloc for Pandas DataFrame
    noisy_points_subset = df.iloc[indices].copy() # Select a subset of ORIGINAL points to *become* noisy

    # Add noise to the *subset* (noisy_points)
    noisy_points_subset['X'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points_subset['Y'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points_subset['Z'] += np.random.normal(0, position_noise_std, num_noisy_points)
    if 'Red' in df.columns and 'Green' in df.columns and 'Blue' in df.columns:
        noisy_points_subset['Red'] = np.clip(noisy_points_subset['Red'] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1)
        noisy_points_subset['Green'] = np.clip(noisy_points_subset['Green'] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1)
        noisy_points_subset['Blue'] = np.clip(noisy_points_subset['Blue'] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1)

    noisy_points_subset['Tag'] = 'Noise' # Tag the *noisy subset* as 'Noise'

    # Concatenate original DataFrame with the *noisy subset*
    df_merged = pd.concat([df, noisy_points_subset], ignore_index=True) # Concatenate with noisy_points_subset
    return df_merged

def process_point(i, points, colors, neighbor_indices, sigma_s, sigma_c):
    indices = neighbor_indices[i]
    distances = np.linalg.norm(points[indices] - points[i], axis=1)
    spatial_weights = np.exp(- (distances ** 2) / (2 * sigma_s ** 2))
    color_diff = np.linalg.norm(colors[indices] - colors[i], axis=1)
    color_weights = np.exp(- (color_diff ** 2) / (2 * sigma_c ** 2))
    weights = spatial_weights * color_weights
    weights /= np.sum(weights)
    filtered_point = np.sum(points[indices] * weights[:, np.newaxis], axis=0)
    filtered_color = np.sum(colors[indices] * weights[:, np.newaxis], axis=0)
    return filtered_point, filtered_color

def bilateral_filter_point_cloud(pcd, num_neighbors=10, sigma_s=0.1, sigma_c=0.1, n_jobs=-1):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # Build a NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', n_jobs=n_jobs)
    nn.fit(points)
    # Get neighbor indices for all points
    _, neighbor_indices = nn.kneighbors(points)

    # Use threading backend to avoid pickling issues
    with parallel_backend('threading'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_point)(i, points, colors, neighbor_indices, sigma_s, sigma_c)
            for i in range(len(points))
        )

    # Unpack results
    filtered_points, filtered_colors = zip(*results)

    # Update the point cloud
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd

def calculate_l2_norm(original_points, filtered_points):
    return np.linalg.norm(original_points - filtered_points, axis=1).mean()

def calculate_l_inf_norm(original_points, filtered_points):
    return np.linalg.norm(original_points - filtered_points, axis=1).max()

def filter_and_evaluate(df_noisy, pcd_filtered, displacement_threshold=0.1, log_file_path="Real/evaluation_log.txt", parameter_setting = "Nothing to show"):
    """
    Evaluate the filtered point cloud and generate evaluation metrics.
    Computes overall binary metrics and provides a breakdown of misclassified real points
    by their original tag, including the percentage of misclassification compared to the
    total number of points from that tag.
    """
    os.makedirs("Real", exist_ok=True)
    start_time = time.time()

    points_filtered = np.asarray(pcd_filtered.points)
    colors_filtered = np.asarray(pcd_filtered.colors)
    df_filtered = df_noisy.copy()

    # Add filtered coordinates and colors
    df_filtered['X_filtered'] = points_filtered[:, 0]
    df_filtered['Y_filtered'] = points_filtered[:, 1]
    df_filtered['Z_filtered'] = points_filtered[:, 2]
    df_filtered['Red_filtered'] = colors_filtered[:, 0]
    df_filtered['Green_filtered'] = colors_filtered[:, 1]
    df_filtered['Blue_filtered'] = colors_filtered[:, 2]

    # Calculate displacement
    displacement = np.linalg.norm(
        df_filtered[['X', 'Y', 'Z']].values - df_filtered[['X_filtered', 'Y_filtered', 'Z_filtered']].values,
        axis=1
    )
    df_filtered['displacement'] = displacement

    # Classify points based on displacement.
    # Points with displacement greater than the threshold are identified as 'Noise'
    # Otherwise, they are marked as 'Real points'
    df_filtered['predicted_tag'] = np.where(displacement > displacement_threshold, 'Noise', 'Real points')

    # Ensure the ground truth is in the same binary space.
    # Any point that is not originally marked as 'Noise' is considered a 'Real point'
    df_filtered['Tag_binary'] = np.where(df_filtered['Tag'] == 'Noise', 'Noise', 'Real points')

    # --- Evaluation block: breakdown of misclassified real points ---
    # We want to know, among the points that are truly real (non-noise)
    # but were misclassified (i.e. predicted as 'Noise'), how many come from each original tag.
    misclassified_real_points = df_filtered[(df_filtered['Tag_binary'] == 'Real points') &
                                            (df_filtered['predicted_tag'] == 'Noise')]
    misclassified_by_original_tag = misclassified_real_points['Tag'].value_counts()

    # Compute total counts for each real tag (ignoring points originally labeled as 'Noise')
    total_real_counts = df_filtered[df_filtered['Tag'] != 'Noise']['Tag'].value_counts()

    # Compute misclassification percentage per original tag
    misclassified_percentage = (misclassified_by_original_tag / total_real_counts * 100).round(2)

    # Compute total number of true noise points in the model (ground truth)
    total_noise_points = df_filtered[df_filtered['Tag'] == 'Noise'].shape[0]

    # Compute the number of noise points correctly predicted (True Positives for noise)
    true_noise_predictions = df_filtered[(df_filtered['Tag'] == 'Noise') & 
                                        (df_filtered['predicted_tag'] == 'Noise')].shape[0]

    # Calculate the ratio (as a percentage, if desired)
    if total_noise_points > 0:
        noise_recall_ratio = true_noise_predictions / total_noise_points
        noise_recall_percentage = noise_recall_ratio * 100
    else:
        noise_recall_ratio = None
        noise_recall_percentage = None

    print(f"Noise correctly predicted: {true_noise_predictions} out of {total_noise_points} noise points, "
        f"which is {noise_recall_ratio:.2f} (or {noise_recall_percentage:.2f}%).")

    # Combine the counts and percentages into a single DataFrame for logging
    misclassified_breakdown = pd.DataFrame({
        'misclassified_count': misclassified_by_original_tag,
        'total_count': total_real_counts,
        'misclassified_percentage': misclassified_percentage
    })

    y_true_pd = df_filtered['Tag_binary']
    y_pred_pd = df_filtered['predicted_tag']

    # Map labels to numeric values for binary metrics
    label_map = {'Real points': 0, 'Noise': 1}
    y_true_numeric = y_true_pd.map(label_map)
    y_pred_numeric = y_pred_pd.map(label_map)

    # Confusion matrix and related metrics
    conf_matrix = confusion_matrix(y_true_numeric, y_pred_numeric, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    precision = precision_score(y_true_numeric, y_pred_numeric, pos_label=1)
    recall = recall_score(y_true_numeric, y_pred_numeric, pos_label=1)
    f1 = f1_score(y_true_numeric, y_pred_numeric, pos_label=1)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Real Points', 'Noise'], yticklabels=['Real Points', 'Noise'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix: Bilateral Filtering")
    confusion_matrix_path = os.path.join("Real", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    total_time = time.time() - start_time

    # Calculate misclassification details (binary summary)
    misclassified_mask = df_filtered['Tag_binary'] != df_filtered['predicted_tag']
    misclassified_points = df_filtered[misclassified_mask]
    total_points = df_filtered.shape[0]
    misclassified_summary = misclassified_points.groupby('Tag_binary').agg({
        'Tag_binary': 'count'
    }).rename(columns={'Tag_binary': 'count'})
    misclassified_summary['percentage'] = misclassified_summary['count'] / total_points * 100

    # Calculate norms
    l2_norm = calculate_l2_norm(df_noisy[['X', 'Y', 'Z']].values, points_filtered)
    l_inf_norm = calculate_l_inf_norm(df_noisy[['X', 'Y', 'Z']].values, points_filtered)

    # Write evaluation results to log file
    with open(log_file_path, 'a') as f:
        f.write(f"Parameter used: {parameter_setting}\n")
        f.write("___________________________________:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
        f.write("Misclassified Points Summary:\n")
        f.write(misclassified_summary.to_string() + "\n")
        f.write(f"Noise correctly predicted: {true_noise_predictions} out of {total_noise_points} noise points, "
        f"which is {noise_recall_ratio:.2f} (or {noise_recall_percentage:.2f}%).\n")
        f.write(f"L2 Norm: {l2_norm:.4f}\n")
        f.write(f"L-Infinity Norm: {l_inf_norm:.4f}\n")
        f.write(f"Processing Time: {total_time:.2f}s\n")
        f.write("\nBreakdown of misclassified real points by original tag:\n")
        f.write(misclassified_by_original_tag.to_string() + "\n")
        f.write("\nBreakdown of misclassified real points by original tag (with percentages):\n")
        f.write(misclassified_breakdown.to_string() + "\n")

    # Save filtered point cloud
    ply_path = os.path.join("Real", "filtered_point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd_filtered)

    return df_filtered


if __name__ == "__main__":
    output_folder = "./Bilateral_Sparse"
    create_output_folder(output_folder)

    file_path = '/data/landfills_UAV/3dData/FinalMesh/Asbestos2.las'
    #file_path = '/data/landfills_UAV/3dData/PointClouds/odm_georeferenced_model_Chiuduno.las'

    num_neighbors_values = [10, 50, 100, 200]
    sigma_s_values = [0.1, 0.5, 1.0, 2.0]
    sigma_c_values = [0.01, 0.1, 0.5, 1.0]

    for num_neighbors in num_neighbors_values:
        for sigma_s in sigma_s_values:
            for sigma_c in sigma_c_values:
                params_folder = os.path.join(output_folder)
                create_output_folder(params_folder)
                log_file = os.path.join(params_folder, "processing_log.txt")
                log_file_eval = os.path.join(params_folder, "evaluation_log.txt")

                log_to_file(log_file, "Starting LAS file processing with parameters:")
                log_to_file(log_file, f"num_neighbors: {num_neighbors}, sigma_s: {sigma_s}, sigma_c: {sigma_c}")

                try:
                    # Import LAS to DataFrame
                    df = import_las_to_dataframe(file_path)
                    log_to_file(log_file, f"Successfully imported LAS file with {df.shape[0]} points.")

                    # Add noise
                    df_noisy = add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=1, color_noise_std=1)
                    log_to_file(log_file, f"Added noise to the data, resulting in {df_noisy.shape[0]} total points.")
                    information = "nn" + str(num_neighbors) + "ss_" + str(sigma_s) + "_sc_" + str(sigma_c)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(df_noisy[['X', 'Y', 'Z']].to_numpy())
                    pcd.colors = o3d.utility.Vector3dVector(df_noisy[['Red', 'Green', 'Blue']].to_numpy())
                    pcd_filtered = bilateral_filter_point_cloud(pcd, num_neighbors=num_neighbors, sigma_s=sigma_s, sigma_c=sigma_c, n_jobs=-1)
                    df_filtered = filter_and_evaluate(df_noisy, pcd_filtered, displacement_threshold=0.1, log_file_path=log_file_eval, parameter_setting=information)

                    log_to_file(log_file, "Finished processing and evaluation for this parameter set.")

                except Exception as e:
                    log_to_file(log_file, f"Error occurred: {str(e)}")
                    raise