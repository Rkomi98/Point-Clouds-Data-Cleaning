import os
import laspy
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed, parallel_backend

def create_output_folder(folder_path):
    """Creates an output folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    """Adds noisy points to a copy of the input DataFrame"""
    num_points = df.shape[0]
    num_noisy_points = int(noise_percentage / 100 * num_points)
    indices = np.random.choice(num_points, num_noisy_points, replace=False)

    noisy_points = df.iloc[indices].copy()
    noisy_points['X'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Y'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Z'] += np.random.normal(0, position_noise_std, num_noisy_points)
    
    if 'Red' in df.columns:
        for channel in ['Red', 'Green', 'Blue']:
            noisy_points[channel] = np.clip(
                noisy_points[channel] + np.random.normal(0, color_noise_std, num_noisy_points),
                0, 1
            )
    
    # Use numeric tags instead of strings
    noisy_points['Tag'] = 255  # Noise flag

    return pd.concat([df, noisy_points], ignore_index=True)

def add_outliers_to_dataframe(df, num_outlier_clusters=4, cluster_size_range=(50, 200),
                              cluster_distance_range=(1, 4), position_noise_std=5.0):
    """Adds synthetic outlier clusters to the DataFrame"""
    min_x, max_x = df['X'].min(), df['X'].max()
    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = df['Z'].min(), df['Z'].max()

    outlier_points = []
    for _ in range(num_outlier_clusters):
        cluster_center = [
            np.random.uniform(min_x - cluster_distance_range[1], max_x + cluster_distance_range[1]),
            np.random.uniform(min_y - cluster_distance_range[1], max_y + cluster_distance_range[1]),
            np.random.uniform(min_z - cluster_distance_range[1], max_z + cluster_distance_range[1])
        ]
        cluster_size = np.random.randint(*cluster_size_range)
        cluster_points = np.random.normal(loc=cluster_center, scale=position_noise_std, size=(cluster_size, 3))
        
        outlier_df = pd.DataFrame(cluster_points, columns=['X', 'Y', 'Z'])
        if 'Red' in df.columns:
            for channel in ['Red', 'Green', 'Blue']:
                outlier_df[channel] = np.random.uniform(0, 1, cluster_size)
        
        # Use numeric tags instead of strings
        outlier_df['Tag'] = 254  # Outlier flag
        
        outlier_points.append(outlier_df)
    
    return pd.concat([df] + outlier_points, ignore_index=True)

def process_point(i, points, colors, neighbor_indices, sigma_s, sigma_c):
    """Process individual points for bilateral filtering"""
    indices = neighbor_indices[i]
    distances = np.linalg.norm(points[indices] - points[i], axis=1)
    spatial_weights = np.exp(-(distances ** 2) / (2 * sigma_s ** 2))
    
    color_diff = np.linalg.norm(colors[indices] - colors[i], axis=1)
    color_weights = np.exp(-(color_diff ** 2) / (2 * sigma_c ** 2))
    
    weights = spatial_weights * color_weights
    weights /= np.sum(weights)
    
    filtered_point = np.sum(points[indices] * weights[:, np.newaxis], axis=0)
    filtered_color = np.sum(colors[indices] * weights[:, np.newaxis], axis=0)
    return filtered_point, filtered_color

def bilateral_filter_point_cloud(pcd, num_neighbors=10, sigma_s=0.1, sigma_c=0.1, n_jobs=-1):
    """Apply bilateral filtering to point cloud"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    nn = NearestNeighbors(n_neighbors=num_neighbors, n_jobs=n_jobs)
    nn.fit(points)
    _, neighbor_indices = nn.kneighbors(points)

    with parallel_backend('threading'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_point)(i, points, colors, neighbor_indices, sigma_s, sigma_c)
            for i in range(len(points))
        )

    filtered_points, filtered_colors = zip(*results)
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    return pcd

def save_las_file(points, colors, tags, original_header, output_path):
    """Save point cloud data to LAS file"""
    header = laspy.LasHeader(
        version=original_header.version,
        point_format=original_header.point_format
    )
    header.scales = original_header.scales
    header.offsets = original_header.offsets

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if colors is not None:
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)

    # Check if tag dimension exists and is of integer type
    if tags is not None and 'tag' in header.point_format.dimension_names:
        # Ensure tags are integer type
        las.tag = tags.astype(np.uint8)

    las.write(output_path)

if __name__ == "__main__":
    output_folder = "./Processed_PointClouds"
    create_output_folder(output_folder)

    # 1. Read and prepare original data
    input_path = "C:/Users/Legion-pc-polimi/Documents/SyntheticDataLidar/Asbestos45.las"
    las_file = laspy.read(input_path)

    # Create DataFrame using verified array access
    df = pd.DataFrame(
        las_file.xyz,
        columns=['X', 'Y', 'Z']
    )

    # Handle colors and tags using proper array access
    has_colors = all(c in las_file.point_format.dimension_names for c in ['red', 'green', 'blue'])
    has_tags = 'tag' in las_file.point_format.dimension_names

    if has_colors:
        df['Red'] = las_file.red / 65535.0
        df['Green'] = las_file.green / 65535.0
        df['Blue'] = las_file.blue / 65535.0

    if has_tags:
        df['Tag'] = las_file.tag
    else:
        df['Tag'] = 0  # Default tag for original points

    # 2. Add noise and outliers
    df_noisy = add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5)
    df_noisy = add_outliers_to_dataframe(df_noisy, num_outlier_clusters=4)

    # 3. Save noisy point cloud
    noisy_colors = df_noisy[['Red', 'Green', 'Blue']].values if has_colors else None
    save_las_file(
        points=df_noisy[['X', 'Y', 'Z']].values,
        colors=noisy_colors,
        tags=df_noisy['Tag'].values if has_tags else None,
        original_header=las_file.header,
        output_path=os.path.join(output_folder, "noisy_point_cloud.las")
    )

    # 4. Process noisy point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df_noisy[['X', 'Y', 'Z']].values)
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(df_noisy[['Red', 'Green', 'Blue']].values)

    # Statistical outlier removal
    sor_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=15, 
        std_ratio=7.0
    )

    # Bilateral filtering
    filtered_pcd = bilateral_filter_point_cloud(
        sor_pcd,
        num_neighbors=50,
        sigma_s=1,
        sigma_c=1
    )

    # Displacement filtering
    original_points = np.asarray(sor_pcd.points)
    filtered_points = np.asarray(filtered_pcd.points)
    displacement = np.linalg.norm(original_points - filtered_points, axis=1)
    valid_mask = displacement <= 1e-9
    print(len(displacement))

    # 5. Save cleaned and removed points
    all_indices = np.arange(len(df_noisy))
    final_cleaned_indices = all_indices[inlier_indices][valid_mask]
    removed_indices = np.setdiff1d(all_indices, final_cleaned_indices)

    # Save cleaned points
    save_las_file(
        points=df_noisy.iloc[final_cleaned_indices][['X', 'Y', 'Z']].values,
        colors=df_noisy.iloc[final_cleaned_indices][['Red', 'Green', 'Blue']].values if has_colors else None,
        tags=df_noisy.iloc[final_cleaned_indices]['Tag'].values if has_tags else None,
        original_header=las_file.header,
        output_path=os.path.join(output_folder, "cleaned_points.las")
    )

    # Save removed points
    save_las_file(
        points=df_noisy.iloc[removed_indices][['X', 'Y', 'Z']].values,
        colors=df_noisy.iloc[removed_indices][['Red', 'Green', 'Blue']].values if has_colors else None,
        tags=df_noisy.iloc[removed_indices]['Tag'].values if has_tags else None,
        original_header=las_file.header,
        output_path=os.path.join(output_folder, "removed_points.las")
    )

    print("Processing complete. Output files:")
    print(f"- Noisy point cloud: {os.path.join(output_folder, 'noisy_point_cloud.las')}")
    print(f"- Cleaned points: {os.path.join(output_folder, 'cleaned_points.las')}")
    print(f"- Removed points: {os.path.join(output_folder, 'removed_points.las')}")