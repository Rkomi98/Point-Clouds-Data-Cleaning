import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import time
import os
from scipy.spatial import KDTree

# --- Output Directories ---
output_dir_meanshift = "MeanShift_Results_Color_Corrected_45" # Output directory for Mean Shift method
os.makedirs(output_dir_meanshift, exist_ok=True)

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

def evaluate_denoising_metrics(original_points, denoised_points, method_name, execution_time, output_dir, k_curvature=15, method_params_str=""): # MODIFIED: Renamed function, removed tag-specific stuff, added k_curvature
    """
    Evaluates denoising performance using metrics from Hu, Peng, Forrest paper.
    """

    # --- 1. Vertex Drifting Norms (L2 and L-infinity) ---
    drift_vectors = denoised_points - original_points # Assuming points are in same order
    l2_norm = np.linalg.norm(drift_vectors, axis=1)
    l_infinity_norm = np.max(drift_vectors)
    mean_l2_norm = np.mean(l2_norm)

    # --- 2. Curvature Analysis (Mean and Variance) ---
    pcd_noisy_eval = o3d.geometry.PointCloud()
    pcd_noisy_eval.points = o3d.utility.Vector3dVector(original_points) # Use original points for noisy curvature
    normals_noisy, curvature_noisy = estimate_normals_and_curvature(pcd_noisy_eval, k_curvature)
    mean_curvature_noisy = np.mean(curvature_noisy)
    variance_curvature_noisy = np.var(curvature_noisy)

    pcd_denoised_eval = o3d.geometry.PointCloud()
    pcd_denoised_eval.points = o3d.utility.Vector3dVector(denoised_points)
    normals_denoised, curvature_denoised = estimate_normals_and_curvature(pcd_denoised_eval, k_curvature)
    mean_curvature_denoised = np.mean(curvature_denoised)
    variance_curvature_denoised = np.var(curvature_denoised)


    print(f"\n--- Denoising Metrics for {method_name} {method_params_str}---") # MODIFIED: Added method_params_str
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Mean L2 Norm of Vertex Drift: {mean_l2_norm:.4f}")
    print(f"L-infinity Norm of Vertex Drift: {l_infinity_norm:.4f}")

    print("\n--- Curvature Statistics (k={}) ---".format(k_curvature))
    print(f"Noisy Point Cloud - Mean Curvature: {mean_curvature_noisy:.4f}, Variance: {variance_curvature_noisy:.4f}")
    print(f"Denoised Point Cloud - Mean Curvature: {mean_curvature_denoised:.4f}, Variance: {variance_curvature_denoised:.4f}")


    metrics_filename = os.path.join(output_dir, f"{method_name}{method_params_str}_denoising_metrics_45.txt") # MODIFIED: Added method_params_str to filename
    with open(metrics_filename, "w") as f:
        f.write(f"Denoising Metrics for {method_name} {method_params_str}\n") # MODIFIED: Added method_params_str to header
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")
        f.write(f"Mean L2 Norm of Vertex Drift: {mean_l2_norm:.4f}\n")
        f.write(f"L-infinity Norm of Vertex Drift: {l_infinity_norm:.4f}\n")
        f.write("\n--- Curvature Statistics (k={}) ---\n".format(k_curvature)) # MODIFIED: Added k_curvature to file output
        f.write(f"Noisy Point Cloud - Mean Curvature: {mean_curvature_noisy:.4f}, Variance: {variance_curvature_noisy:.4f}\n")
        f.write(f"Denoised Point Cloud - Mean Curvature: {mean_curvature_denoised:.4f}, Variance: {variance_curvature_denoised:.4f}\n")


    return mean_l2_norm, l_infinity_norm, execution_time, mean_curvature_noisy, variance_curvature_noisy, mean_curvature_denoised, variance_curvature_denoised

df_merged = add_noise_to_dataframe(df)
# Corrected line: Use parentheses for each condition and then bitwise OR
y_true_original = np.where((df_merged['Tag'] == 'Noise'), 1, 0) #(df_merged['Tag'] == 'Outlier') |

# --- Fixed sigma parameters for trilateral filter ---
sigma_spatial = 0.5
sigma_normal = 0.1
sigma_curvature = 0.05

# --- Add Mean Shift Denoising Functions ---

def estimate_normals_and_curvature(pcd, k):
    """
    Estimate normals and curvature for each point using PCA.
    """
    points = np.asarray(pcd.points)
    normals_est = np.zeros((len(points), 3))
    curvature = np.zeros(len(points))

    kdtree = KDTree(points)

    for i in range(len(points)):
        # Find k-NN
        neighbor_indices = kdtree.query(points[i], k=k+1)[1][1:]  # Exclude self
        neighbors = points[neighbor_indices]

        if len(neighbors) < 3:
            normals_est[i] = np.array([0, 0, 0])
            curvature[i] = 0
            continue

        # PCA for normal and curvature
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

        # Normal is eigenvector corresponding to smallest eigenvalue
        normals_est[i] = eigenvectors[:, 0]
        normals_est[i] /= np.linalg.norm(normals_est[i])

        # Curvature is λ1 / (λ1 + λ2 + λ3)
        eigenvalue_sum = np.sum(eigenvalues)
        if eigenvalue_sum > 1e-6: # Avoid division by zero
            curvature[i] = eigenvalues[0] / eigenvalue_sum
        else:
            curvature[i] = 0.0 # Set curvature to 0 if sum is too small
    curvature = (curvature - np.min(curvature)) / \
                (np.max(curvature) - np.min(curvature) + 1e-8)

    return normals_est, curvature

def compute_mean_shift_vector(point_i, neighbors, h_range_normal, h_range_curvature, h_range_color, has_colors): # Added has_colors flag
    shift_vector = np.zeros(3)
    total_weight = 0

    normal_i = point_i[3:6]
    curvature_i = point_i[6]
    if has_colors: # Check the flag here
        color_i = point_i[7:10] # Get color if available

    for neighbor in neighbors:
        spatial_diff = neighbor[:3] - point_i[:3]
        normal_diff = np.linalg.norm(neighbor[3:6] - normal_i)
        curvature_diff = abs(neighbor[6] - curvature_i)

        # Gaussian kernels
        w_normal = np.exp(-0.5 * (normal_diff / h_range_normal)**2)
        w_curv = np.exp(-0.5 * (curvature_diff / h_range_curvature)**2)
        combined_weight = w_normal * w_curv

        if has_colors: # Check the flag here
            color_neighbor = neighbor[7:10] # Get neighbor color
            color_diff = np.linalg.norm(color_neighbor - color_i)
            w_color = np.exp(-0.5 * (color_diff / h_range_color)**2)
            combined_weight *= w_color # Multiply with color weight


        shift_vector += combined_weight * spatial_diff
        total_weight += combined_weight  # FIXED: Single accumulation

    if total_weight > 1e-8:
        shift_vector /= total_weight

    return shift_vector

def cluster_local_modes(shifted_points, cell_size=0.01):
    """
    Cluster points with similar local modes using octree.
    """
    pcd_shifted = o3d.geometry.PointCloud()
    pcd_shifted.points = o3d.utility.Vector3dVector(shifted_points[:, :3])

    # Create octree and insert points
    octree = o3d.geometry.Octree(max_depth=8)
    octree.convert_from_point_cloud(pcd_shifted, size_expand=cell_size)

    # Extract indices of points in each leaf node
    clustered_neighbors = []

    def traverse_callback(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            clustered_neighbors.append(node.indices)
        return False  # Continue traversal

    # Traverse the octree
    octree.traverse(traverse_callback)

    return clustered_neighbors

def trilateral_filter(pcd, clustered_neighbors, normals_est, curvature, sigma_spatial=sigma_spatial, sigma_normal=sigma_normal, sigma_curvature=sigma_curvature):
    """
    Apply trilateral filtering to the point cloud using clusters.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(normals_est) # Use pre-calculated normals, corrected line
    denoised_points = np.copy(points)

    point_to_cluster = {}
    # Add intensity normalization
    max_dist = np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1))
    spatial_scale = 0.1 * max_dist  # Adaptive spatial scale
    for cluster_idx, cluster_indices in enumerate(clustered_neighbors):
        for point_index in cluster_indices:
            point_to_cluster[point_index] = cluster_idx

    for i in range(len(points)):
        cluster_index = point_to_cluster.get(i)
        if cluster_index is None: # Point might not be assigned to any cluster (unlikely with current clustering)
            continue
        neighbor_indices_in_cluster = clustered_neighbors[cluster_index]
        weights = []
        weighted_sum = np.zeros(3)

        for j_index_in_cluster in neighbor_indices_in_cluster: # corrected line: use index from cluster
            j = j_index_in_cluster # corrected line: get original point index
            if i == j: # Skip self-reference
                continue
            if i >= len(points) or j >= len(points) or i >= len(normals) or j >= len(normals) or i < 0 or j < 0: # Added bounds check
                continue

            dist = np.linalg.norm(points[i] - points[j])
            normal_diff = np.arccos(np.clip(np.dot(normals[i], normals[j]), -1, 1))
            curv_diff = abs(curvature[i] - curvature[j])

            w_spatial = np.exp(-(dist**2) / (2 * (sigma_spatial * spatial_scale)**2))
            w_normal = np.exp(-normal_diff**2 / (2 * sigma_normal**2))
            w_curv = np.exp(-curv_diff**2 / (2 * sigma_curvature**2))
            total_weight = w_spatial * w_normal * w_curv

            weighted_sum += points[j] * total_weight
            weights.append(total_weight)

        if sum(weights) > 1e-8:
            denoised_points[i] = weighted_sum / sum(weights)

    return denoised_points

def mean_shift_denoising(pcd_noisy, k_spatial=15, h_range=(0.02, 0.9), h_range_color=0.1, max_iter=10, epsilon=0.001, sigma_spatial=sigma_spatial, sigma_normal=sigma_normal, sigma_curvature=sigma_curvature, has_colors=False): # Pass has_colors flag
    """
    Perform mean shift denoising on a point cloud, preserving colors.
    """
    start_time = time.time()
    noisy_colors = None
    if pcd_noisy.has_colors():
        noisy_colors = np.asarray(pcd_noisy.colors)
        has_colors = True # Set flag to True if colors are present


    # Step 1: Estimate normals and curvature
    normals_est, curvature = estimate_normals_and_curvature(pcd_noisy, k_spatial) # Get normals_est

    points = np.asarray(pcd_noisy.points)

    # Step 2: Build generalized points [x, y, z, nx, ny, nz, curvature, R, G, B]
    if noisy_colors is not None:
        generalized_points = np.hstack((points, normals_est, curvature.reshape(-1, 1), noisy_colors)) # Include colors
    else:
        generalized_points = np.hstack((points, normals_est, curvature.reshape(-1, 1))) # Without colors


    # Step 3: Mean shift iterations
    shifted_points = np.copy(generalized_points)
    kdtree = KDTree(points)

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        converged = True

        for i in range(len(shifted_points)):
            point_i = shifted_points[i]

            # --- Adaptive Neighbor Search ---
            # 1. Spatial neighbors
            spatial_neighbor_indices = kdtree.query(point_i[:3], k=k_spatial)[1][1:]  # Exclude self
            spatial_neighbors = generalized_points[spatial_neighbor_indices]

            # 2. Range neighbors (filter by normal/curvature bandwidths)
            normal_diff = np.linalg.norm(spatial_neighbors[:, 3:6] - point_i[3:6], axis=1)
            curvature_diff = np.abs(spatial_neighbors[:, 6] - point_i[6])
            range_mask = (normal_diff < h_range[0]) & (curvature_diff < h_range[1])
            if noisy_colors is not None: # Add color range check if colors are available
                color_diff = np.linalg.norm(spatial_neighbors[:, 7:10] - point_i[7:10], axis=1) # Assuming colors start at index 7
                range_mask = range_mask & (color_diff < h_range_color) # Combine with color range

            adaptive_neighbors = spatial_neighbors[range_mask]

            # --- Compute Mean Shift Vector ---
            if noisy_colors is not None:
                shift_vector = compute_mean_shift_vector(point_i, adaptive_neighbors, h_range[0], h_range[1], h_range_color, has_colors) # Pass h_range_color and has_colors
            else:
                shift_vector = compute_mean_shift_vector(point_i, adaptive_neighbors, h_range[0], h_range[1], h_range_color=0, has_colors=False) # Pass dummy h_range_color and has_colors=False

            # --- Update Position ---
            if np.linalg.norm(shift_vector) > epsilon:
                shifted_points[i, :3] += shift_vector
                converged = False

        if converged:
            print("Mean shift converged!")
            break

    # Step 4: Clustering local modes
    clustered_neighbors = cluster_local_modes(shifted_points)

    # Step 5: Trilateral filtering
    denoised_points = trilateral_filter(pcd_noisy, clustered_neighbors, normals_est, curvature, sigma_spatial, sigma_normal, sigma_curvature) # Pass pcd_noisy, normals_est

    execution_time = time.time() - start_time
    print(f"Denoising completed in {execution_time:.2f} seconds.")

    return denoised_points, execution_time


# --- Apply Mean Shift Denoising to Noisy Data and Parameter Sweep ---

df_noisy = df_merged[['X', 'Y', 'Z', 'Red', 'Green', 'Blue']].copy() # Keep a copy of noisy points WITH colors for evaluation
original_points_for_eval = df_noisy[['X', 'Y', 'Z']].values # Use the noisy points as "original" for drift calculation


k_spatial_values = [15] # Reduced k_spatial values for faster testing
h_range_color_values = [0.05, 0.1, 0.2] # Test different color bandwidths

results = [] # List to store results for each k

for k_spatial in k_spatial_values:
    for h_range_color in h_range_color_values:
        print(f"\n--- Running Mean Shift Denoising with k_spatial = {k_spatial}, h_range_color = {h_range_color} ---")

        # Convert noisy DataFrame to Open3D point cloud
        pcd_noisy = o3d.geometry.PointCloud()
        pcd_noisy.points = o3d.utility.Vector3dVector(df_noisy[['X', 'Y', 'Z']].values) # Pass XYZ
        if colors is not None:
            pcd_noisy.colors = o3d.utility.Vector3dVector(df_noisy[['Red', 'Green', 'Blue']].values) # Pass noisy colors from df_noisy

        # Run mean shift denoising
        denoised_points, execution_time = mean_shift_denoising(
            pcd_noisy, k_spatial=k_spatial, h_range=(0.02, 0.9), h_range_color=h_range_color, max_iter=10, epsilon=0.001, sigma_spatial=sigma_spatial, sigma_normal=sigma_normal, sigma_curvature=sigma_curvature # Pass sigma parameters
        )

        # Save denoised point cloud with color
        pcd_denoised = o3d.geometry.PointCloud()
        pcd_denoised.points = o3d.utility.Vector3dVector(denoised_points)
        if colors is not None:
            pcd_denoised.colors = o3d.utility.Vector3dVector(df_noisy[['Red', 'Green', 'Blue']].values) # Use noisy colors for denoised point cloud!
        output_filename = os.path.join(output_dir_meanshift, f"denoised_point_cloud_k{k_spatial}_color{h_range_color}.ply")
        o3d.io.write_point_cloud(output_filename, pcd_denoised, write_ascii=True, compressed=False)
        print(f"Denoised point cloud with k={k_spatial} and color bandwidth={h_range_color} saved to: {output_filename}")


        method_params_str = f"(k_spatial={k_spatial}, h_range={(0.02, 0.9)}, h_range_color={h_range_color})"
        mean_l2, l_inf, exec_time, mean_curv_noisy, var_curv_noisy, mean_curv_denoised, var_curv_denoised = evaluate_denoising_metrics(
            original_points_for_eval, denoised_points, "Mean Shift Denoising", execution_time, output_dir_meanshift, k_curvature=15, method_params_str=method_params_str
        )

        results.append({ # Store results in the list
            'k_spatial': k_spatial,
            'h_range_color': h_range_color,
            'mean_l2_drift': mean_l2,
            'l_inf_drift': l_inf,
            'execution_time': exec_time,
            'mean_curvature_noisy': mean_curv_noisy,
            'variance_curvature_noisy': var_curv_noisy,
            'mean_curvature_denoised': mean_curv_denoised,
            'variance_curvature_denoised': var_curv_denoised
        })

        print(f"Denoising Metrics (k={k_spatial}, color_bw={h_range_color}): Mean L2 Drift={mean_l2:.4f}, L-inf Drift={l_inf:.4f}, Exec Time={exec_time:.4f}")
        print(f"Curvature (Noisy, k=15): Mean={mean_curv_noisy:.4f}, Var={var_curv_noisy:.4f}")
        print(f"Curvature (Denoised, k=15): Mean={mean_curv_denoised:.4f}, Var={var_curv_denoised:.4f}")

# --- Summarize Results Table ---
print("\n--- Parameter Sweep Results Summary ---")
results_df = pd.DataFrame(results)
print(results_df)

results_table_filename = os.path.join(output_dir_meanshift, "parameter_sweep_results_table_color_45.txt")
with open(results_table_filename, "w") as f:
    f.write("Parameter Sweep Results Summary (Color Bandwidth):\n")
    f.write(results_df.to_string())

print(f"\nParameter sweep results table saved to: {results_table_filename}")