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
    Enhanced evaluation of results with detailed classification breakdown,
    including per-class accuracy.

    Parameters:
    - df_combined: DataFrame with original ('Tag') and predicted ('predicted_tag') tags.
    - log_file_path: Path to log file for detailed results.
    - displacement_threshold: Threshold used for displacement classification (for logging).
    - parameter_setting: String describing parameter settings used.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    start_time = time.time()

    # Create a copy to avoid modifying the original DataFrame passed to the function
    df_eval = df_combined.copy()

    # Ensure predicted_tag exists and fill NaNs if any (shouldn't happen with current logic, but good practice)
    if 'predicted_tag' not in df_eval.columns:
        df_eval['predicted_tag'] = 'Unknown'
    df_eval['predicted_tag'].fillna('Unknown', inplace=True) # Fill potential NaNs

    # Prepare true and predicted labels
    y_true = df_eval['Tag']
    y_pred = df_eval['predicted_tag']

    # Define all unique classes present in either true or predicted labels
    all_classes = sorted(list(pd.unique(df_eval[['Tag', 'predicted_tag']].values.ravel('K'))))
    # Ensure 'Unknown' is handled if it appears
    if 'Unknown' in all_classes and 'Unknown' not in df_eval['Tag'].unique():
         # Add 'Unknown' to true labels list if only in predicted, for consistent matrix shape
         pass # No need to add if confusion_matrix handles it via labels=


    # Compute confusion matrix using all potential classes
    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_classes)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)

    # Compute row-wise percentages (Recall for each class)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore division by zero warnings
        conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        conf_matrix_percentages = np.nan_to_num(conf_matrix_percentages) # Replace NaN with 0
    conf_matrix_percentages = np.round(conf_matrix_percentages, 2)
    percentage_matrix_df = pd.DataFrame(conf_matrix_percentages, index=all_classes, columns=all_classes)

    # Compute standard classification report
    # Use zero_division=0 to handle classes with no predicted samples gracefully
    class_report = classification_report(y_true, y_pred, labels=all_classes, zero_division=0)

    # --- Calculate Per-Class Accuracy ---
    per_class_accuracy = {}
    total_samples = len(y_true)
    for i, cls in enumerate(all_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = total_samples - TP - FP - FN

        if (TP + TN + FP + FN) == 0:
            accuracy = 0.0 # Or np.nan, depending on desired representation
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        per_class_accuracy[cls] = round(accuracy * 100, 2) # Store as percentage

    # --- Detailed Breakdown (as before) ---
    detailed_breakdown = {}
    original_classes = sorted(df_eval['Tag'].unique().tolist())
    for original_class in original_classes:
        class_points = df_eval[df_eval['Tag'] == original_class]
        total_class_points = len(class_points)
        if total_class_points == 0:
             detailed_breakdown[original_class] = {
                'total_points': 0,
                'prediction_counts': {},
                'prediction_percentages': {}
            }
             continue

        prediction_counts = class_points['predicted_tag'].value_counts()
        prediction_percentages = (prediction_counts / total_class_points * 100).round(2)
        detailed_breakdown[original_class] = {
            'total_points': total_class_points,
            'prediction_counts': prediction_counts.to_dict(),
            'prediction_percentages': prediction_percentages.to_dict()
        }

    # --- Visualization (as before) ---
    plt.figure(figsize=(max(12, len(all_classes)), max(10, len(all_classes)))) # Adjust size dynamically
    combined_matrix_annot = np.zeros_like(conf_matrix_df, dtype=object)
    for i, r_cls in enumerate(all_classes):
        for j, c_cls in enumerate(all_classes):
            combined_matrix_annot[i, j] = f"{conf_matrix[i, j]}\n({conf_matrix_percentages[i, j]:.2f}%)"

    sns.heatmap(percentage_matrix_df, annot=combined_matrix_annot, fmt="", cmap="Blues",
                xticklabels=all_classes, yticklabels=all_classes, vmin=0, vmax=100) # Use percentage matrix for color scale
    plt.title(f"Confusion Matrix (Count and Row Percentage)\nParameters: {parameter_setting}\nDisplacement Threshold: {displacement_threshold}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    confusion_matrix_path = os.path.join(os.path.dirname(log_file_path), "detailed_confusion_matrix.png")
    try:
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")
    plt.close()


    # --- Logging results ---
    with open(log_file_path, 'w') as f:
        f.write(f"=== Detailed Evaluation ===\n")
        f.write(f"Parameter Setting: {parameter_setting}\n")
        f.write(f"Displacement Threshold: {displacement_threshold}\n")
        f.write(f"Evaluation Time: {time.time() - start_time:.2f} seconds\n\n")

        f.write("--- Confusion Matrix (Counts) ---\n")
        f.write(conf_matrix_df.to_string() + "\n\n")

        f.write("--- Confusion Matrix (Row Percentages) ---\n")
        f.write(percentage_matrix_df.to_string() + "\n\n")

        f.write("--- Classification Report (Precision, Recall, F1-Score) ---\n")
        f.write(class_report + "\n\n")

        f.write("--- Per-Class Accuracy (%) ---\n")
        for cls, acc in per_class_accuracy.items():
            f.write(f"  {cls}: {acc:.2f}%\n")
        f.write("\n")

        f.write("--- Detailed Classification Breakdown (True Label -> Predicted) ---\n")
        for original_class, breakdown in detailed_breakdown.items():
            f.write(f"{original_class} (Total: {breakdown['total_points']}):\n")
            # Sort predictions for consistent output
            sorted_preds = sorted(breakdown['prediction_counts'].items())
            for pred_class, count in sorted_preds:
                percentage = breakdown['prediction_percentages'].get(pred_class, 0)
                f.write(f"    -> {pred_class}: {count} points ({percentage:.2f}%)\n")
            f.write("\n")

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
    output_folder = "./Bilateral_SORBF_D/Aggressive"
    create_output_folder(output_folder)

    file_path = 'C:/Users/Legion-pc-polimi/Documents/SyntheticDataLidar/Asbestos45.las'
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

    # --- REVISED LOGIC for predicted_tag assignment using np.select ---

    # Define the conditions
    conditions = [
        # Case 1: Low displacement, original is Noise -> Unlabelled
        (df_sor_inliers['displacement'] <= DISPLACEMENT_THRESHOLD) & (df_sor_inliers['Tag'] == 'Noise'),

        # Case 2: High displacement (regardless of original tag) -> Noise
        (df_sor_inliers['displacement'] > DISPLACEMENT_THRESHOLD)
    ]

    # Define the corresponding choices for each condition
    choices = [
        'Vegetation', # Choice for condition 1
        'Noise'       # Choice for condition 2
    ]

    # Define the default choice (handles low displacement, original NOT Noise)
    # In this case, the default is to keep the original tag.
    default_choice = df_sor_inliers['Tag']

    # Apply np.select
    df_sor_inliers['predicted_tag'] = np.select(conditions, choices, default=default_choice)
    #TODO Cambiare 'Real Points' con df_sor_inliers['Tag'] o simile. Inoltre aggiornare CM con i tag presenti nella PC

    # Combine all points for evaluation
    df_combined = pd.concat([df_sor_inliers, df_sor_outliers], ignore_index=True)

    # --- Step 3: Calculate Displacement and Assign Predicted Tags ---
    print("Calculating displacement and assigning predicted tags...")

    # Ensure alignment before calculating displacement
    if len(points_filtered) == len(df_sor_inliers):
        displacement = np.linalg.norm(
            df_sor_inliers[['X', 'Y', 'Z']].values - points_filtered,
            axis=1
        )
        df_sor_inliers['displacement'] = displacement
    else:
        print(f"Warning: Mismatch in point count after bilateral filter ({len(points_filtered)}) and SOR inliers DataFrame ({len(df_sor_inliers)}). Skipping displacement calculation.")
        df_sor_inliers['displacement'] = np.nan # Indicate missing displacement


    # Assign predicted tags based on displacement and original tag
    # Define conditions for np.select
    conditions = [
        # Condition 1: High displacement -> Predicted as Noise
        (df_sor_inliers['displacement'] > DISPLACEMENT_THRESHOLD),
        # Condition 2: Low displacement AND original tag was Noise -> Reclassify (e.g., to Unclassified or Terrain/Vegetation)
        # Let's reclassify originally noisy points with low displacement as their original non-noise neighbors might be.
        # A simple approach is to default them to 'Unclassified' or perhaps a common class like 'Terrain' or 'Vegetation'.
        # Reclassifying to 'Vegetation' as per original code snippet:
        (df_sor_inliers['displacement'] <= DISPLACEMENT_THRESHOLD) & (df_sor_inliers['Tag'] == 'Noise')
        # Add more specific conditions if needed
    ]

    # Define choices corresponding to conditions
    choices = [
        'Noise',        # Choice for Condition 1
        'Vegetation'    # Choice for Condition 2 (Example reclassification)
    ]

    # Default choice: If none of the above conditions are met (i.e., low displacement and NOT originally Noise),
    # keep the original tag.
    default_choice = df_sor_inliers['Tag']

    # Apply np.select
    df_sor_inliers['predicted_tag'] = np.select(conditions, choices, default=default_choice)

    # Handle potential NaN displacements if calculation failed
    df_sor_inliers['predicted_tag'] = df_sor_inliers['predicted_tag'].fillna('Unknown_Displacement_Issue')

    # Combine all points (SOR inliers with predictions + SOR outliers)
    df_combined = pd.concat([df_sor_inliers, df_sor_outliers], ignore_index=True)
    print("Predicted Tag Counts (Before Final Filtering):")
    print(df_combined['predicted_tag'].value_counts())


    # --- Step 4: Evaluate Full Results (Optional but Recommended) ---
    print("Evaluating classification performance on all points...")
    evaluate_combined_results(
        df_combined,
        log_file_eval,
        displacement_threshold=DISPLACEMENT_THRESHOLD,
        parameter_setting=f"SOR: {SOR_PARAMS}, Bilateral: {BILATERAL_PARAMS}"
    )

    # --- Step 5: Filter out predicted Noise and Outliers for Clean Export ---
    print("Filtering out points predicted as Noise or Outlier...")
    df_cleaned = df_combined[
        ~df_combined['predicted_tag'].isin(['Noise', 'Outlier'])
    ].copy() # Use .copy() to be safe
    print(f"Points remaining after filtering: {len(df_cleaned)}")
    print("Tag Counts in Cleaned Data (Using Original Tags):")
    print(df_cleaned['Tag'].value_counts()) # Show original tags of the cleaned points

    # --- Step 6: Export the Cleaned Point Cloud ---
    # This export will contain only points NOT predicted as Noise/Outlier.
    # The 'classification' field in the LAS file will be based on the *original* 'Tag'.
    cleaned_las_path = os.path.join(output_folder, "point_cloud_cleaned_NoNoiseOutliers.las")
    print(f"Exporting cleaned point cloud (no predicted Noise/Outliers) to: {cleaned_las_path}")
    export_point_cloud(df_cleaned, cleaned_las_path) # Reuse the export function


    # --- Step 7: Export Full Point Cloud with Predictions (Optional) ---
    # This export includes ALL points and adds the 'predicted_tag' as an extra dimension.
    print("Exporting full point cloud with predicted tags as extra dimension...")

    # Remap string predicted tags to numeric for the extra dimension
    # Use the inverse mapping derived from the original tag_mapping
    # Add Noise and Outlier to the mapping if they don't exist
    predicted_tag_numeric_mapping = inverse_tag_mapping.copy()
    if 'Noise' not in predicted_tag_numeric_mapping:
        predicted_tag_numeric_mapping['Noise'] = 7 # Assign a number
    if 'Outlier' not in predicted_tag_numeric_mapping:
        predicted_tag_numeric_mapping['Outlier'] = 8 # Assign a number
    if 'Unknown_Displacement_Issue' not in predicted_tag_numeric_mapping:
         predicted_tag_numeric_mapping['Unknown_Displacement_Issue'] = 9 # Assign a number


    # Convert predicted tags to numeric values using the extended map
    # Map, fill NaN, then infer the best possible dtype (which should be numeric here)
    predicted_numeric = df_combined['predicted_tag'].map(predicted_tag_numeric_mapping).fillna(0)
    # Explicitly convert to the desired final type
    df_combined['predicted_tag_numeric'] = predicted_numeric.infer_objects(copy=False).astype(np.uint8, copy=False)


    # Create a new LAS header based on the original, adding the extra dimension
    new_header = laspy.LasHeader(
        version=las_file.header.version,
        point_format=las_file.header.point_format
    )
    # Copy scales and offsets IS IMPORTANT
    new_header.scales = las_file.header.scales
    new_header.offsets = las_file.header.offsets

    # Check if the point format already supports the desired dimensions, else add extra bytes
    # Standard classification field name is 'classification'
    has_classification_field = 'classification' in new_header.point_format.dimension_names
    
    # Check if 'predicted_tag' could be an existing extra dim
    has_predicted_tag_field = "predicted_tag" in new_header.point_format.dimension_names

    if not has_predicted_tag_field:
        try:
            new_header.add_extra_dim(laspy.ExtraBytesParams(
                name="predicted_tag",
                type=np.uint8, # Ensure type matches your data
                description="Pred Tag (7=N, 8=O, 9=U)" 
            ))
            print("Added 'predicted_tag' as an extra dimension.")
        except Exception as e:
            print(f"Error adding extra dimension 'predicted_tag': {e}")
            print("Proceeding without extra dimension for predicted tag.")


    # Create new LAS data object
    new_las = laspy.LasData(new_header)

    # Populate standard dimensions from df_combined
    new_las.x = df_combined['X'].values
    new_las.y = df_combined['Y'].values
    new_las.z = df_combined['Z'].values

    if 'Red' in df_combined.columns:
        # Ensure colors are scaled back to uint16 for LAS
        new_las.red = (np.clip(df_combined['Red'].values, 0, 1) * 65535).astype(np.uint16)
        new_las.green = (np.clip(df_combined['Green'].values, 0, 1) * 65535).astype(np.uint16)
        new_las.blue = (np.clip(df_combined['Blue'].values, 0, 1) * 65535).astype(np.uint16)

    # Populate original classification field (using 'classification' standard name)
    if has_classification_field:
         # Map original 'Tag' string back to numeric using inverse_tag_mapping
        df_combined['original_tag_numeric'] = df_combined['Tag'].map(inverse_tag_mapping).fillna(0).astype(np.uint8) # Default to 0 for unknown
        new_las.classification = df_combined['original_tag_numeric'].values
    elif 'tag' in new_header.point_format.dimension_names: # Fallback to 'tag' if present
        df_combined['original_tag_numeric'] = df_combined['Tag'].map(inverse_tag_mapping).fillna(0).astype(np.uint8)
        new_las.tag = df_combined['original_tag_numeric'].values


    # Populate the predicted_tag extra dimension (if successfully added)
    if "predicted_tag" in new_las.point_format.dimension_names:
         # Ensure the name matches exactly how it was added or exists
         setattr(new_las, "predicted_tag", df_combined['predicted_tag_numeric'].values)
    else:
         print("Skipping population of 'predicted_tag' field as it's not available in the header.")


    # Write the output LAS file with all points and predictions
    output_las_path_with_predictions = os.path.join(output_folder, "classified_with_predictions.las")
    print(f"Writing final LAS file with predictions to: {output_las_path_with_predictions}")
    new_las.write(output_las_path_with_predictions)
    print("Processing complete.")

    