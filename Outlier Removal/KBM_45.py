import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 
import time
from sklearn.preprocessing import StandardScaler

las_file = laspy.read("/data/landfills_UAV/3dData/FinalMesh/Asbestos45.las")
# Extract point data (X, Y, Z coordinates)
# Laspy stores coordinates as scaled integers, so you must scale them appropriately
points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()

fmt = las_file.point_format
# Extract color information if available (colors are stored as 16-bit integers in LAS)
if 'red' in las_file.point_format.dimension_names:
    red = las_file.red / 65535.0  # Scale from 16-bit to 0-1 range
    green = las_file.green / 65535.0
    blue = las_file.blue / 65535.0
    colors = np.vstack((red, green, blue)).transpose()
    print("Colors!")
else:
    colors = None  # If the LAS file doesn't have color data
    print("No Colors!")
    
# Extract tag (extra field) if it exists in the file
if fmt.dimension_by_name('tag'):
    #fmt.dimension_by_name('tag')
    tags_numeric = las_file.tag
    print("Tags!")
else:
    tags = None  # If the LAS file doesn't have the 'tag' field
    print("No tags :-(")

# Create the DataFrame with X, Y, Z coordinates
df = pd.DataFrame(data=points, 
                  columns=['X','Y','Z'])  # 1st row as the column names
df['Red'] = red
df['Green'] = green
df['Blue'] = blue

# Sample tag mapping
tag_mapping = {
    "Vegetation": 1,
    "Terrain": 2,
    "Metals": 3,
    "Asbestos": 4,
    "Tyres": 5,
    "Plastics": 6,
    "default": 0  # Default if no tag is provided
}

# Reverse the tag_mapping to map numbers back to tags
inverse_tag_mapping = {v: k for k, v in tag_mapping.items()}
# Use numpy vectorization to map numeric values back to tags
vectorized_mapping = np.vectorize(inverse_tag_mapping.get)

# Apply the mapping to the array
tags = vectorized_mapping(tags_numeric)

df['Tag'] = tags

#%% New dataset tag creation
all_tags = df['Tag'].unique()

# Define the tags we want to handle separately
separate_tags = ['Vegetation', 'Terrain']

# Create dictionaries to store our point variables and coordinates
waste_points = {}
waste_coords = {}

# Handle terrain separately to get the minimum Z-coordinate
terrain_points = df[df['Tag'] == 'Terrain']
vegetation_points = df[df['Tag'] == 'Vegetation']
terrain_coords = terrain_points[['X', 'Y', 'Z']].values
veg_coords = vegetation_points[['X', 'Y', 'Z']].values
min_terrain_z = np.min(terrain_coords[:, 2])
max_terrain_z = np.max(veg_coords[:, 2])

df = df[df['Z'] >= min_terrain_z]

def add_noise_to_dataframe(df, noise_percentage=10, position_noise_std=0.5, color_noise_std=0.05):
    """
    Adds noisy points near the surface and optionally introduces noise to color features (RGB).
    
    Parameters:
    - df: The original DataFrame with point cloud data.
    - noise_percentage: Percentage of points to generate noise for (default is 5%).
    - position_noise_std: Standard deviation for generating positional noise (default is 0.01).
    - color_noise_std: Standard deviation for generating RGB noise (default is 0.05).
    
    Returns:
    - df_merged: The merged DataFrame with original and noisy points.
    """

    # Number of points to generate noise for
    num_points = df.shape[0]
    num_noisy_points = int(noise_percentage / 100 * num_points)
    
    # Step 1: Generate noisy points near the surface
    # Select random indices for the noisy points
    indices = np.random.choice(df.index, num_noisy_points, replace=False)

    # Extract the original points to be perturbed
    noisy_points = df.loc[indices, ['X', 'Y', 'Z']].copy()

    # Add noise to the X, Y, Z coordinates
    noisy_points['X'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Y'] += np.random.normal(0, position_noise_std, num_noisy_points)
    noisy_points['Z'] += np.random.normal(0, position_noise_std, num_noisy_points)
    
    # Step 2: Add noise to the color features (Red, Green, Blue)
    if 'Red' in df.columns and 'Green' in df.columns and 'Blue' in df.columns:
        noisy_points['Red'] = df.loc[indices, 'Red'] + np.random.normal(0, color_noise_std, num_noisy_points)
        noisy_points['Green'] = df.loc[indices, 'Green'] + np.random.normal(0, color_noise_std, num_noisy_points)
        noisy_points['Blue'] = df.loc[indices, 'Blue'] + np.random.normal(0, color_noise_std, num_noisy_points)
        
        # Ensure color values remain in the valid range [0, 1]
        noisy_points['Red'] = noisy_points['Red'].clip(0, 1)
        noisy_points['Green'] = noisy_points['Green'].clip(0, 1)
        noisy_points['Blue'] = noisy_points['Blue'].clip(0, 1)
    else:
        # If the original data doesn't have color, set color to None
        noisy_points['Red'] = np.nan
        noisy_points['Green'] = np.nan
        noisy_points['Blue'] = np.nan

    # Step 3: Assign 'Noise' to the Tag column for these noisy points
    noisy_points['Tag'] = 'Noise'

    # Step 4: Combine the noisy points with the original DataFrame
    df_merged = pd.concat([df, noisy_points], ignore_index=True)

    return df_merged
def add_outliers_to_dataframe(df, num_outlier_clusters=4, cluster_size_range=(50, 200), 
                              cluster_distance_range=(1, 4), position_noise_std=5.0):
    """
    Adds outlier clusters to the DataFrame to simulate erroneous data.
    
    Parameters:
    - df: The original DataFrame with point cloud data.
    - num_outlier_clusters: Number of outlier clusters to generate (default is 5).
    - cluster_size_range: Tuple specifying the min and max number of points per cluster (default is (50, 200)).
    - cluster_distance_range: Tuple specifying the min and max distance from the original point cloud (default is (1, 4)).
    - position_noise_std: Standard deviation for adding noise to the outlier positions (default is 5.0).
    
    Returns:
    - df_merged: The DataFrame with original and outlier points.
    """
    # Get the bounding box of the original data
    min_x, max_x = df['X'].min(), df['X'].max()
    min_y, max_y = df['Y'].min(), df['Y'].max()
    min_z, max_z = min_terrain_z, max_terrain_z

    # Create outlier clusters
    outlier_points = []
    for _ in range(num_outlier_clusters):
        # Randomly choose a central position for the outlier cluster
        cluster_center = [
            np.random.uniform(min_x - cluster_distance_range[1], max_x + cluster_distance_range[1]),
            np.random.uniform(min_y - cluster_distance_range[1], max_y + cluster_distance_range[1]),
            np.random.uniform(min_z - cluster_distance_range[1], max_z + cluster_distance_range[1])
        ]
        
        # Randomly choose the number of points in the cluster
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        
        # Generate points around the cluster center with added noise
        cluster_points = np.random.normal(
            loc=cluster_center,
            scale=position_noise_std,
            size=(int(cluster_size), 3)
        )
        
        # Create a DataFrame for the outlier points
        outlier_df = pd.DataFrame(cluster_points, columns=['X', 'Y', 'Z'])
        outlier_df['Tag'] = 'Outlier'
        
        # Optionally, add random colors to the outlier points
        outlier_df['Red'] = np.random.uniform(0, 1, cluster_size)
        outlier_df['Green'] = np.random.uniform(0, 1, cluster_size)
        outlier_df['Blue'] = np.random.uniform(0, 0.01, cluster_size)
        
        outlier_points.append(outlier_df)
    
    # Combine the outlier points with the original DataFrame
    outlier_df_combined = pd.concat(outlier_points, ignore_index=True)
    df_merged = pd.concat([df, outlier_df_combined], ignore_index=True)
    
    return df_merged 
def indices_confusion_matrix(y_true, y_pred, method_name, execution_time, original_tags):
    """
    Plots the confusion matrix with 'Outlier' and 'Not Outlier' labels, and exports metrics and the plot.

    Parameters:
    - y_true: Ground truth labels (1 for 'Outlier', 0 for 'Not Outlier').
    - y_pred: Predicted labels (1 for 'Outlier', 0 for 'Not Outlier').
    - method_name: A string indicating the name of the method, used in file naming.
    """
    # Step 1: Replace 0 with 'Not Outlier' and 1 with 'Outlier' for readability
    label_mapping = {0: 'Not Outlier', 1: 'Outlier'}
    y_true_str = [label_mapping[label] for label in y_true]
    y_pred_str = [label_mapping[label] for label in y_pred]

    # Step 2: Generate confusion matrix
    conf_matrix = confusion_matrix(y_true_str, y_pred_str, labels=['Not Outlier', 'Outlier'])
    tn, fp, fn, tp = conf_matrix.ravel()  # Confusion matrix layout: [[TN, FP], [FN, TP]]
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate

    # Step 3: Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(y_true_str, y_pred_str)
    precision = precision_score(y_true_str, y_pred_str, pos_label='Outlier')
    recall = recall_score(y_true_str, y_pred_str, pos_label='Outlier')
    f1 = f1_score(y_true_str, y_pred_str, pos_label='Outlier')

    # Step 5: Analyze misclassified tags
    misclassified_mask = (y_true != y_pred)
    misclassified_tags = original_tags[misclassified_mask]
    
    # Count tag occurrences for misclassified points
    tag_counts = pd.Series(misclassified_tags).value_counts()
    
    print("\nMisclassified Tag Distribution:")
    print(tag_counts)

    # Save metrics to a text file
    metrics_filename = f"{method_name}_metrics_45.txt"
    with open(metrics_filename, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
        f.write(f"Execution Time: {execution_time}")
        f.write("\nMisclassified Tag Distribution:\n")
        f.write(tag_counts.to_string())

    # Print the performance metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    # Step 4: Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Outlier', 'Outlier'], yticklabels=['Not Outlier', 'Outlier'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix: {method_name}")

    # Save the confusion matrix plot to a PNG file
    plot_filename = f"{method_name}_confusion_matrix_45.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return tpr, fpr

df_merged = add_noise_to_dataframe(df)
df_merged = add_outliers_to_dataframe(df_merged)

kernel = 'rbf' #kmeans' # We can use many other function, it is the one of the paper
gamma = None  # default value

start_time = time.time()
# Here we consider two spectral clustering because we have outliers vs real object

clustering = SpectralClustering(n_clusters=2,
                                affinity=kernel,
                                assign_labels='cluster_qr',#initialization is not so important, I hope
                                #assign_labels='kmeans',  # Try instead of 'cluster_qr'
                                random_state=0,
                                verbose = True,
                                n_neighbors=50,)
def create_sampled_dataset(df_merged, sampling_rate=50,outlier_sampling_rate=1):#500
    """
    Samples the DataFrame for all points except the 'Outlier' points, then returns features (X) and binary labels (y).
    
    Parameters:
    - df_merged: DataFrame containing the columns ['X', 'Y', 'Z', 'Red', 'Green', 'Blue', 'Tag'].
    - sampling_rate: Integer, specifies the sampling rate for non-Outlier points.

    Returns:
    - X: NumPy array of shape (n_samples, 6), containing features (X, Y, Z, Red, Green, Blue).
    - y: NumPy array of shape (n_samples,), binary labels (1 for Outlier, 0 for others).
    """
    
    # Step 1: Separate 'Outlier' points (keep all)
    outliers = df_merged[df_merged['Tag'] == 'Outlier'].iloc[::outlier_sampling_rate]
    
    # Step 2: Sample all other points (Tag != 'Outlier') every `sampling_rate` rows
    non_outliers = df_merged[df_merged['Tag'] != 'Outlier'].iloc[::sampling_rate]
    
    # Step 3: Combine sampled non-Outlier points with all Outlier points
    df_sampled = pd.concat([non_outliers, outliers], ignore_index=True)

    # Step 4: Create X (features) and y (binary labels)
    X = df_sampled[['X', 'Y', 'Z', 'Red', 'Green', 'Blue']].to_numpy()

    # Convert 'Outlier' to 1, all other tags to 0
    y = np.where(df_sampled['Tag'] == 'Outlier', 1, 0)
    
    return X, y, df_sampled

# Example usage:
X, y, df_sampled = create_sampled_dataset(df_merged)

X_scaled = StandardScaler().fit_transform(X)
clusters = clustering.fit(X_scaled)
#clusters = clustering.fit(X)

predicted_labels = clusters.labels_
end_time = time.time()

# Compute the time taken
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")

if np.sum(predicted_labels == 0) < np.sum(predicted_labels == 1):
    predicted_labels = 1 - predicted_labels  # Invert labels

indices_confusion_matrix(y, predicted_labels, 'Kernel Based method', execution_time, df_sampled['Tag'])


