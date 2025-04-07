import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import laspy
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
    las = laspy.read(file_path)
    
    # Extract required fields and convert to numpy arrays
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    r = np.array(las.red) / 65535  # Normalize to [0, 1]
    g = np.array(las.green) / 65535  # Normalize to [0, 1]
    b = np.array(las.blue) / 65535  # Normalize to [0, 1]
    
    # Create Polars DataFrame
    df = pl.DataFrame({
        'X': x,
        'Y': y,
        'Z': z,
        'Red': r,
        'Green': g,
        'Blue': b,
        'Tag': ['Real points'] * len(x)
    })
    
    return df

def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    """
    Adds noisy points near the surface and optionally introduces noise to color features (RGB).
    """
    num_points = df.shape[0]
    num_noisy_points = int(noise_percentage / 100 * num_points)
    
    # Randomly select indices for noisy points
    indices = np.random.choice(num_points, num_noisy_points, replace=False)
    # Convert indices to a list of integers
    indices = indices.tolist()
    # Select rows by row index
    selected_points = df.with_row_index(name="row_id").filter(pl.col("row_id").is_in(indices)).to_pandas()
    
    # Add noise to positional and color data
    noisy_points = pl.DataFrame({
        "X": selected_points["X"] + np.random.normal(0, position_noise_std, num_noisy_points),
        "Y": selected_points["Y"] + np.random.normal(0, position_noise_std, num_noisy_points),
        "Z": selected_points["Z"] + np.random.normal(0, position_noise_std, num_noisy_points),
        "Red": np.clip(selected_points["Red"] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1),
        "Green": np.clip(selected_points["Green"] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1),
        "Blue": np.clip(selected_points["Blue"] + np.random.normal(0, color_noise_std, num_noisy_points), 0, 1),
        "Tag": pl.Series(['Noise'] * num_noisy_points)
    })
    
    # Concatenate original and noisy points
    df_merged = pl.concat([df, noisy_points])
    return df_merged
def add_outliers_to_dataframe(df, num_outlier_clusters=4, cluster_size_range=(50, 200), 
                               cluster_distance_range=(1, 4), position_noise_std=5.0):
    """
    Adds outlier clusters to the DataFrame to simulate erroneous data.
    """
    min_x, max_x = df["X"].min(), df["X"].max()
    min_y, max_y = df["Y"].min(), df["Y"].max()
    min_z, max_z = df["Z"].min(), df["Z"].max()
    
    outlier_points = []
    for _ in range(num_outlier_clusters):
        cluster_center = [
            np.random.uniform(min_x - cluster_distance_range[1], max_x + cluster_distance_range[1]),
            np.random.uniform(min_y - cluster_distance_range[1], max_y + cluster_distance_range[1]),
            np.random.uniform(min_z - cluster_distance_range[1], max_z + cluster_distance_range[1])
        ]
        cluster_size = np.random.randint(*cluster_size_range)
        
        cluster_data = np.random.normal(loc=cluster_center, scale=position_noise_std, size=(cluster_size, 3))
        outlier_df = pl.DataFrame({
            'X': cluster_data[:, 0],
            'Y': cluster_data[:, 1],
            'Z': cluster_data[:, 2],
            'Red': np.random.uniform(0, 1, cluster_size),
            'Green': np.random.uniform(0, 1, cluster_size),
            'Blue': np.random.uniform(0, 1, cluster_size),
            'Tag': ['Outlier'] * cluster_size
        })
        outlier_points.append(outlier_df)
    
    # Concatenate original and outlier points
    outlier_df_combined = pl.concat(outlier_points)
    df_merged = pl.concat([df, outlier_df_combined])
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
    '''
    with parallel_backend('threading'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_point)(i, points, colors, neighbor_indices, sigma_s, sigma_c)
            for i in range(len(points))
        )
    '''
    results = Parallel(n_jobs=n_jobs)(delayed(process_point)(i, points, colors, neighbor_indices, sigma_s, sigma_c) for i in range(len(points)))
    
    # Unpack results
    filtered_points, filtered_colors = zip(*results)
    
    # Update the point cloud
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    return pcd

def filter_and_evaluate(df_noisy, pcd_filtered, displacement_threshold=0.1, log_file_path="Synth/evaluation_log.txt"):
    # Ensure the Output folder exists
    os.makedirs("Synth", exist_ok=True)
    # Start timing
    start_time = time.time()

    points_filtered = np.asarray(pcd_filtered.points)
    colors_filtered = np.asarray(pcd_filtered.colors)
    df_filtered = df_noisy.clone()

    # Add filtered coordinates and colors
    df_filtered = df_filtered.with_columns([
        pl.Series('X_filtered', points_filtered[:, 0]),
        pl.Series('Y_filtered', points_filtered[:, 1]),
        pl.Series('Z_filtered', points_filtered[:, 2]),
        pl.Series('Red_filtered', colors_filtered[:, 0]),
        pl.Series('Green_filtered', colors_filtered[:, 1]),
        pl.Series('Blue_filtered', colors_filtered[:, 2])
    ])

    # Compute displacement
    displacement = np.linalg.norm(
        df_filtered[['X', 'Y', 'Z']].to_numpy() - df_filtered[['X_filtered', 'Y_filtered', 'Z_filtered']].to_numpy(),
        axis=1
    )
    df_filtered = df_filtered.with_columns(pl.Series('displacement', displacement))

    # Add predicted tags
    df_filtered = df_filtered.with_columns(
        pl.when(pl.col('displacement') > displacement_threshold)
          .then(pl.lit('Anomaly'))  # Combine Noise and Outliers as "Anomaly"
          .otherwise(pl.lit('Real points'))
          .alias('predicted_tag')
    )

    # Debug: Check final DataFrame
    print("Final columns in df_filtered:", df_filtered.columns)
    print(df_filtered.head(5))

    # Calculate total counts for each tag
    tag_counts = df_noisy['Tag'].value_counts().to_dict()
    
    y_true_pd = df_filtered['Tag'].to_pandas()
    y_pred_pd = df_filtered['predicted_tag'].to_pandas()
    
    # Map labels to numeric values
    #label_map = {'Real points': 0, 'Noise': 1}
    # Map labels to numeric values
    y_true_numeric = y_true_pd.map({'Real points': 0, 'Noise': 1, 'Outlier': 1})
    y_pred_numeric = y_pred_pd.map({'Real points': 0, 'Anomaly': 1})  # Map "Anomaly" to 
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true_numeric, y_pred_numeric, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    precision = precision_score(y_true_numeric, y_pred_numeric, pos_label=1)
    recall = recall_score(y_true_numeric, y_pred_numeric, pos_label=1)
    f1 = f1_score(y_true_numeric, y_pred_numeric, pos_label=1)
    
    # Save confusion matrix as PNG
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Real Points', 'Noise'], yticklabels=['Real Points', 'Noise'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix: Bilateral Filtering")
    confusion_matrix_path = os.path.join("Synth", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    total_time = time.time() - start_time
    # Identify false negatives and false positives
    fn_points = df_filtered.filter((pl.col('Tag') == 'Noise') & (pl.col('predicted_tag') == 'Real points'))
    #misclassified_tags_fn = printPercentageMisclassifedV(fn_points, tag_counts)
    
    fp_points = df_filtered.filter((pl.col('Tag') == 'Real points') & (pl.col('predicted_tag') == 'Noise'))
    #misclassified_tags_fp = printPercentageMisclassifedF(fp_points, tag_counts)
    
    # Write metrics and confusion matrix path to the log file
    with open(log_file_path, 'a') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
        f.write(f"It took {total_time}s \n")
    
    return df_filtered

if __name__ == "__main__":
    output_folder = "./Synth"
    create_output_folder(output_folder)
    log_file = os.path.join(output_folder, "processing_log.txt")
    
    file_path = '/data/landfills_UAV/3dData/FinalMesh/Asbestos2.las'
    #file_path = '/data/landfills_UAV/3dData/PointClouds/odm_georeferenced_model_Chiuduno.las'
    
    log_to_file(log_file, "Starting LAS file processing...")
    
    try:
        # Import LAS to DataFrame
        df = import_las_to_dataframe(file_path)
        log_to_file(log_file, f"Successfully imported LAS file with {df.shape[0]} points.")
        
        # Add noise
        df_noisy = add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=1, color_noise_std=1)
        df_with_outliers = add_outliers_to_dataframe(
            df_noisy, 
            num_outlier_clusters=4, 
            cluster_size_range=(50, 200), 
            cluster_distance_range=(1, 4), 
            position_noise_std=5.0
        )
        
        log_to_file(log_file, f"Added noise to the data, resulting in {df_noisy.shape[0]} total points.")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df_with_outliers[['X', 'Y', 'Z']].to_numpy())
        pcd.colors = o3d.utility.Vector3dVector(df_with_outliers[['Red', 'Green', 'Blue']].to_numpy())
        pcd_filtered = bilateral_filter_point_cloud(pcd, num_neighbors=100, sigma_s=1, sigma_c=1, n_jobs=-1)
        df_filtered = filter_and_evaluate(df_with_outliers, pcd_filtered, displacement_threshold=0.1)
        
    except Exception as e:
        log_to_file(log_file, f"Error occurred: {str(e)}")
        raise