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
from sklearn.metrics import confusion_matrix, classification_report
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
    Enhanced evaluation of results with detailed classification breakdown
    
    Parameters:
    - df_combined: DataFrame with original and predicted tags
    - log_file_path: Path to log file for detailed results
    - displacement_threshold: Threshold for displacement classification
    - parameter_setting: String describing parameter settings used
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    start_time = time.time()

    # Define all possible original point classes
    original_classes = df_combined['Tag'].unique().tolist()
    prediction_classes = original_classes #+ ['Noise', 'Outlier']

    # Create a copy of the DataFrame to avoid modifying the original
    df_eval = df_combined.copy()

    # Ensure predicted_tag exists and is filled
    if 'predicted_tag' not in df_eval.columns:
        df_eval['predicted_tag'] = 'Unknown'

    # Prepare true and predicted labels
    y_true = df_eval['Tag']
    y_pred = df_eval['predicted_tag']

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=prediction_classes)

    # Compute row-wise percentages
    conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    conf_matrix_percentages = np.round(conf_matrix_percentages, 2)

    # Compute classification report
    class_report = classification_report(y_true, y_pred, labels=prediction_classes)

    # Compute detailed classification breakdown
    detailed_breakdown = {}
    for original_class in original_classes:
        # Filter points of this original class
        class_points = df_eval[df_eval['Tag'] == original_class]
        total_class_points = len(class_points)
        
        # Count predictions for this class
        prediction_counts = class_points['predicted_tag'].value_counts()
        
        # Calculate percentages
        prediction_percentages = (prediction_counts / total_class_points * 100).round(2)
        
        detailed_breakdown[original_class] = {
            'total_points': total_class_points,
            'prediction_counts': prediction_counts.to_dict(),
            'prediction_percentages': prediction_percentages.to_dict()
        }

    # Visualization of Confusion Matrix
    plt.figure(figsize=(12, 10))
    
    # Create a combined heatmap with absolute numbers and percentages
    combined_matrix = np.zeros((len(prediction_classes), len(prediction_classes)), dtype=object)
    for i in range(len(prediction_classes)):
        for j in range(len(prediction_classes)):
            combined_matrix[i, j] = f"{conf_matrix[i, j]}\n({conf_matrix_percentages[i, j]:.2f}%)"
    
    sns.heatmap(conf_matrix, annot=combined_matrix, fmt="", cmap="Blues", 
                xticklabels=prediction_classes, yticklabels=prediction_classes)
    plt.title("Confusion Matrix: Detailed Point Classification\n(Count and Row Percentage)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(os.path.dirname(log_file_path), "detailed_confusion_matrix.png")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Logging results
    with open(log_file_path, 'w') as f:
        f.write(f"=== Detailed Evaluation for Parameters: {parameter_setting} ===\n\n")
        
        # Log Confusion Matrix
        f.write("--- Confusion Matrix ---\n")
        conf_matrix_df = pd.DataFrame(conf_matrix, 
                                      index=prediction_classes, 
                                      columns=prediction_classes)
        percentage_matrix_df = pd.DataFrame(
            data=conf_matrix_percentages, 
            index=prediction_classes, 
            columns=prediction_classes
        )
        f.write(conf_matrix_df.to_string() + "\n\n")
        f.write("Percentages (%):\n")
        f.write(percentage_matrix_df.to_string() + "\n\n")
        
        # Log Classification Report
        f.write("--- Classification Report ---\n")
        f.write(class_report + "\n\n")
        
        # Log Detailed Breakdown
        f.write("--- Detailed Classification Breakdown ---\n")
        for original_class, breakdown in detailed_breakdown.items():
            f.write(f"{original_class}:\n")
            f.write(f"  Total Points: {breakdown['total_points']}\n")
            f.write("  Prediction Breakdown:\n")
            for pred_class, count in breakdown['prediction_counts'].items():
                percentage = breakdown['prediction_percentages'].get(pred_class, 0)
                f.write(f"    {pred_class}: {count} points ({percentage}%)\n")
            f.write("\n")

    # Print paths for confirmation
    print(f"Detailed evaluation logged to: {log_file_path}")
    print(f"Confusion Matrix visualization saved to: {confusion_matrix_path}")

    return detailed_breakdown
def export_point_cloud(df, file_path):
    """
    Exports a DataFrame to a LAS file.
    """
    header = laspy.LasHeader(point_format=3, version='1.4')
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]
    
    las = laspy.LasData(header)
    las.x = df['X'].values
    las.y = df['Y'].values
    las.z = df['Z'].values
    
    if 'Red' in df.columns:
        las.red = (df['Red'].values * 65535).astype(np.uint16)
        las.green = (df['Green'].values * 65535).astype(np.uint16)
        las.blue = (df['Blue'].values * 65535).astype(np.uint16)
    
    if 'Tag' in df.columns:
        tag_mapping = {"Vegetation": 1, "Terrain": 2, "Metals": 3, "Asbestos": 4, "Tyres": 5, "Plastics": 6, "Noise": 7, "Outlier": 8}
        las.classification = df['Tag'].map(tag_mapping).fillna(0).astype(np.uint8)
    
    las.write(file_path)
    print(f"Exported point cloud to {file_path}")

if __name__ == "__main__":
    output_folder = "./Bilateral_SORBF_S/Aggressive"
    create_output_folder(output_folder)

    file_path = 'C:/Users/Legion-pc-polimi/Documents/SyntheticDataLidar/Asbestos2.las'
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
    # Export noisy data
    noisy_las_path = os.path.join(output_folder, "noisy_point_cloud.las")
    export_point_cloud(df_noisy, noisy_las_path)

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
        df_sor_inliers['Tag']
    )
    #TODO Cambiare 'Real Points' con df_sor_inliers['Tag'] o simile. Inoltre aggiornare CM con i tag presenti nella PC

    # Combine all points for evaluation
    df_combined = pd.concat([df_sor_inliers, df_sor_outliers], ignore_index=True)

    # --- Step 3: Evaluate Results ---
    evaluate_combined_results(
        df_combined,
        log_file_eval,
        displacement_threshold=DISPLACEMENT_THRESHOLD,
        parameter_setting=f"SOR: {SOR_PARAMS}, Bilateral: {BILATERAL_PARAMS}"
    )

    predicted_tag_mapping = {
        "Vegetation": 1,
        "Terrain": 2,
        "Metals": 3,
        "Asbestos": 4,
        "Tyres": 5,
        "Plastics": 6,
        "Noise": 7,
        "Outlier": 8,
        "Unknown": 0  # Default for unmapped tags
    }

    # Convert predicted tags to numeric values
    df_combined['predicted_tag_numeric'] = df_combined['predicted_tag'].map(predicted_tag_mapping).fillna(0).astype(np.uint8)

    # Create a new LAS header with extra dimension for predicted tag
    new_header = laspy.LasHeader(
        version=las_file.header.version,
        point_format=las_file.header.point_format
    )
    new_header.scales = las_file.header.scales
    new_header.offsets = las_file.header.offsets
    
    # Add predicted_tag as an extra dimension if not already present
    if not hasattr(new_header, 'predicted_tag'):
        new_header.add_extra_dim(laspy.ExtraBytesParams(
            name="predicted_tag",
            type=np.uint8,
            description="Predicted classification tag"
        ))

    # Create new LAS file
    new_las = laspy.LasData(new_header)
    new_las.x = df_combined['X'].values
    new_las.y = df_combined['Y'].values
    new_las.z = df_combined['Z'].values

    # Handle color channels if present
    if 'Red' in df_combined.columns:
        new_las.red = (df_combined['Red'].values * 65535).astype(np.uint16)
        new_las.green = (df_combined['Green'].values * 65535).astype(np.uint16)
        new_las.blue = (df_combined['Blue'].values * 65535).astype(np.uint16)

    # Preserve original classification if available
    if 'tag' in new_las.point_format.dimension_names:
        original_tag_mapping = {v: k for k, v in tag_mapping.items()}
        df_combined['original_tag_numeric'] = df_combined['Tag'].map(tag_mapping).fillna(0).astype(np.uint8)
        new_las.tag = df_combined['original_tag_numeric'].values

    # Add predicted tags
    new_las.predicted_tag = df_combined['predicted_tag_numeric'].values

    # Write output file
    output_las_path = os.path.join(output_folder, "classified_with_predictions.las")
    new_las.write(output_las_path)
    print(f"Exported LAS file with predictions to: {output_las_path}")

    