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

def evaluate_combined_results(df_combined, log_file_path, displacement_threshold=0.1, parameter_setting="None"):
    """
    Evaluates the combined results of SOR and Bilateral Filtering, including:
    - Confusion matrices (multi-class and binary)
    - Misclassification breakdown for real points by original tag
    - Recall for Outlier (SOR) and Noise (Bilateral)
    - Visualization of the confusion matrix
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    start_time = time.time()

    # --- 1. Prepare Ground Truth and Predictions ---
    df_combined['Tag_binary'] = np.where(
        df_combined['Tag'].isin(['Outlier', 'Noise']),
        'Anomaly',
        'Real points'
    )
    df_combined['predicted_binary'] = np.where(
        df_combined['predicted_tag'].isin(['Outlier', 'Noise']),
        'Anomaly',
        'Real points'
    )

    # --- 2. Multi-Class Metrics (Outlier, Noise, Real) ---
    labels = ['Outlier', 'Noise', 'Real points']
    y_true = df_combined['Tag']
    y_pred = df_combined['predicted_tag']
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # --- 3. Recall for Outlier (SOR) and Noise (Bilateral) ---
    outlier_true = (y_true == 'Outlier')
    outlier_pred = (y_pred == 'Outlier')
    noise_true = (y_true == 'Noise')
    noise_pred = (y_pred == 'Noise')

    outlier_recall = np.sum(outlier_true & outlier_pred) / np.sum(outlier_true) if np.sum(outlier_true) > 0 else 0
    noise_recall = np.sum(noise_true & noise_pred) / np.sum(noise_true) if np.sum(noise_true) > 0 else 0

    # --- 4. Misclassified Real Points Breakdown ---
    # Points that are truly real but predicted as anomalies
    misclassified_real_mask = (df_combined['Tag_binary'] == 'Real points') & \
                              (df_combined['predicted_binary'] == 'Anomaly')
    misclassified_real_points = df_combined[misclassified_real_mask]

    # Group by original tag (e.g., Vegetation, Terrain)
    misclassified_by_tag = misclassified_real_points['Tag'].value_counts()
    total_real_by_tag = df_combined[df_combined['Tag_binary'] == 'Real points']['Tag'].value_counts()
    misclassified_percentage = (misclassified_by_tag / total_real_by_tag * 100).fillna(0).round(2)
    end_time = time.time()
    time_needed = start_time - end_time

    # --- 5. Binary Metrics ---
    accuracy = accuracy_score(df_combined['Tag_binary'], df_combined['predicted_binary'])
    precision = precision_score(df_combined['Tag_binary'], df_combined['predicted_binary'], pos_label='Anomaly')
    recall = recall_score(df_combined['Tag_binary'], df_combined['predicted_binary'], pos_label='Anomaly')
    f1 = f1_score(df_combined['Tag_binary'], df_combined['predicted_binary'], pos_label='Anomaly')

    # --- 6. Confusion Matrix Visualization ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix: Outlier, Noise, Real Points")
    confusion_matrix_path = os.path.join(os.path.dirname(log_file_path), "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # --- 7. Logging ---
    with open(log_file_path, 'a') as f:
        f.write(f"\n\n=== Evaluation for Parameters: {parameter_setting} ===\n")
        f.write(f"Time needed (Bilateral): {time_needed:.4f}\n")
        # Multi-class metrics
        f.write("\n--- Multi-Class Metrics ---\n")
        f.write(f"Outlier Recall (SOR): {outlier_recall:.4f}\n")
        f.write(f"Noise Recall (Bilateral): {noise_recall:.4f}\n")
        f.write("Confusion Matrix (Rows: True, Columns: Predicted):\n")
        f.write(pd.DataFrame(conf_matrix, index=labels, columns=labels).to_string() + "\n")
        
        # Binary metrics
        f.write("\n--- Binary Metrics (Anomaly vs Real) ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        
        # Misclassification breakdown
        f.write("\n--- Misclassified Real Points ---\n")
        f.write("Breakdown by Original Tag:\n")
        breakdown_df = pd.DataFrame({
            'Total Points': total_real_by_tag,
            'Misclassified Count': misclassified_by_tag,
            'Misclassified %': misclassified_percentage
        }).fillna(0)
        f.write(breakdown_df.to_string() + "\n")

        # Confusion matrix plot path
        f.write(f"\nConfusion Matrix saved to: {confusion_matrix_path}\n")

    print(f"Full evaluation logged to: {log_file_path}")
    print(f"Confusion Matrix saved to: {confusion_matrix_path}")


if __name__ == "__main__":
    output_folder = "./Bilateral_SORBF_S"
    create_output_folder(output_folder)

    file_path = '/data/landfills_UAV/3dData/FinalMesh/Asbestos2.las'
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
    df_noisy = add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=1, color_noise_std=1)
    df_noisy = add_outliers_to_dataframe(df_noisy)

    #file_path = '/data/landfills_UAV/3dData/PointClouds/odm_georeferenced_model_Chiuduno.las'

    # Define parameters
    SOR_PARAMS = {"nb_neighbors": 15, "std_ratio": 7.0}
    BILATERAL_PARAMS = {"num_neighbors": 10, "sigma_s": 1.0, "sigma_c": 1.0}
    DISPLACEMENT_THRESHOLD = 0.1  # Adjust as needed

    # Create log files
    log_file = os.path.join(output_folder, "processing_log.txt")
    log_file_eval = os.path.join(output_folder, "evaluation_log.txt")
    
    # Load data into Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df_noisy[['X', 'Y', 'Z']].values)
    pcd.colors = o3d.utility.Vector3dVector(df_noisy[['Red', 'Green', 'Blue']].values)

    # --- Step 1: Apply SOR for Outlier Removal ---
    cl, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_PARAMS["nb_neighbors"],
        std_ratio=SOR_PARAMS["std_ratio"]
    )
    sor_inlier_pcd = pcd.select_by_index(inlier_indices)
    sor_outlier_pcd = pcd.select_by_index(inlier_indices, invert=True)

    # Split DataFrame into inliers and outliers
    df_sor_inliers = df_noisy.iloc[inlier_indices].copy()
    df_sor_outliers = df_noisy.iloc[~df_noisy.index.isin(inlier_indices)].copy()
    df_sor_outliers['predicted_tag'] = 'Outlier'  # Tag SOR-removed points

    # --- Step 2: Apply Bilateral Filter to SOR Inliers ---
    sor_inlier_pcd = bilateral_filter_point_cloud(
        sor_inlier_pcd,
        num_neighbors=BILATERAL_PARAMS["num_neighbors"],
        sigma_s=BILATERAL_PARAMS["sigma_s"],
        sigma_c=BILATERAL_PARAMS["sigma_c"],
        n_jobs=-1
    )

    # Compute displacement for inliers after Bilateral
    points_filtered = np.asarray(sor_inlier_pcd.points)
    displacement = np.linalg.norm(
        df_sor_inliers[['X', 'Y', 'Z']].values - points_filtered,
        axis=1
    )
    df_sor_inliers['displacement'] = displacement
    df_sor_inliers['predicted_tag'] = np.where(
        displacement > DISPLACEMENT_THRESHOLD,
        'Noise',
        'Real points'
    )

    # Combine all points for evaluation
    df_combined = pd.concat([df_sor_inliers, df_sor_outliers], ignore_index=True)

    # --- Step 3: Evaluate Results ---
    evaluate_combined_results(
        df_combined,
        log_file_eval,
        displacement_threshold=DISPLACEMENT_THRESHOLD,
        parameter_setting=f"SOR: {SOR_PARAMS}, Bilateral: {BILATERAL_PARAMS}"
    )

    