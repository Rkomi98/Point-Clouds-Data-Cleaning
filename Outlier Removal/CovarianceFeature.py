import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from Preprocessing_funct import preprocessing
from sklearn.decomposition import PCA


las_file = laspy.read("./data/landfills_UAV/3dData/FinalMesh/Asbestos2") ## da cambiare

df_merged = preprocessing(las_file)

# count occurrences of tag 'A'
tag_counts = df_merged['Tag'].value_counts()

# Print the counts for each tag
print(tag_counts)

def indices_confusion_matrix(y_true, y_pred, method_name, save_path='confusion_matrix.png'):
    """
    Plots the confusion matrix with 'Outlier' and 'Not Outlier' labels.
    
    Parameters:
    - y_true: Ground truth labels (1 for 'Outlier', 0 for 'Not Outlier').
    - y_pred: Predicted labels (1 for 'Outlier', 0 for 'Not Outlier').
    """
    # Step 1: Generate confusion matrix using numeric labels 0 and 1
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Using 0 for 'Not Outlier', 1 for 'Outlier'
    
    # Step 2: Extract true negatives, false positives, false negatives, and true positives
    tn, fp, fn, tp = conf_matrix.ravel()  # Confusion matrix layout: [[TN, FP], [FN, TP]]
    
    # Step 3: Calculate TPR and FPR
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate

    # Step 4: Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)  # Use numeric labels for precision
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    # Print the performance metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

    # Step 5: Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Outlier', 'Outlier'], yticklabels=['Not Outlier', 'Outlier'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix: {method_name}")
    plt.show()
    
    # Save the figure as a PNG file
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to avoid displaying it when running on a server
    
    return tpr, fpr

# Assuming you have the true labels `y` and predicted labels `predicted_labels`
# indices_confusion_matrix(y, predicted_labels)


def printPercentageMisclassifedV(misclassified_outliers):
    misclassified_tags = misclassified_outliers['Tag'].value_counts()
    for tag, count in misclassified_tags.items():
        total_count = tag_counts.get(tag, 1)  # Get the total count of the tag, default to 1 to avoid division by zero
        percentage = (count / total_count) * 100  # Calculate percentage
        print(f"{tag}: {count} removed points on {total_count} ({percentage:.2f}%)")
    return misclassified_tags

def printPercentageMisclassifedF(misclassified_outliers):
    misclassified_tags = misclassified_outliers['Tag'].value_counts()
    for tag, count in misclassified_tags.items():
        total_count = tag_counts.get(tag, 1)  # Get the total count of the tag, default to 1 to avoid division by zero
        percentage = (count / total_count) * 100  # Calculate percentage
        print(f"{tag}: {count} not removed points on {total_count} ({percentage:.2f}%)")
    return misclassified_tags

def compute_covariance_matrix(points):
    """ Computes the covariance matrix for a set of points. """
    centered_points = points - np.mean(points, axis=0)
    return np.cov(centered_points, rowvar=False)

# Function to get neighbors within the optimal radius for a given point
def get_neighbors_within_radius(index, radius):
    nbrs = NearestNeighbors(radius=radius).fit(xyz)
    neighbors = nbrs.radius_neighbors([xyz[index]], return_distance=False)[0]
    return neighbors
def compute_shannon_entropy(eigenvalues):
    l1, l2, l3 = eigenvalues
    total = l1 + l2 + l3
    p1 = l1 / total
    p2 = l2 / total
    p3 = l3 / total
    
    entropy = - (p1 * np.log(p1 + 1e-9) + p2 * np.log(p2 + 1e-9) + p3 * np.log(p3 + 1e-9))
    return entropy

def compute_planarity_for_cluster(cluster_points):
    """
    Compute planarity for a cluster based on its eigenvalues.
    Planarity is computed as: (lambda2 - lambda3) / lambda1
    """
    # Cluster points: an array of 3D points (X, Y, Z)
    # Compute the covariance matrix
    cov_matrix = compute_covariance_matrix(cluster_points)
    
    # Perform PCA to get the eigenvalues
    pca = PCA(n_components=3)
    pca.fit(cov_matrix)
    eigenvalues = pca.explained_variance_

    # Ensure eigenvalues are sorted in descending order
    l1, l2, l3 = sorted(eigenvalues, reverse=True)
    
    # Planarity formula
    planarity = (l2 - l3) / l1
    
    return planarity
def compute_anisotropy_for_cluster(cluster_points):
    """
    Compute anisotropy for a cluster based on its eigenvalues.
    Anisotropy is computed as: (lambda1 - lambda3) / lambda1
    """
    # If there are not enough points to compute covariance, return 0 anisotropy
    if len(cluster_points) < 3:
        return 0.0
    
    # Compute the covariance matrix
    cov_matrix = compute_covariance_matrix(cluster_points)
    
    # Check if the covariance matrix is valid (not all zeroes)
    if np.all(cov_matrix == 0):
        return 0.0
    
    # Perform PCA to get the eigenvalues (ensure 2D array input)
    pca = PCA(n_components=3)
    try:
        pca.fit(cov_matrix)
    except ValueError as e:
        print(f"Error in PCA fitting: {e}")
        return 0.0
    
    eigenvalues = pca.explained_variance_

    # Ensure eigenvalues are sorted in descending order
    l1, l2, l3 = sorted(eigenvalues, reverse=True)

    # Anisotropy formula
    anisotropy = (l1 - l3) / l1
    
    return anisotropy


xyz = df_merged[['X', 'Y', 'Z']].values
tags = df_merged['Tag'].values

k = 10  # Adjust k as needed
# Define a range of radii for the neighborhood search
# r_min = 0.1  # Define based on your scene
# r_max = 6.0  # Define based on your scene
# num_radii = 3  # Test 60 different radii as described in the paper
# radii = np.linspace(r_min, r_max, num_radii)
nbrs = NearestNeighbors(n_neighbors=k).fit(xyz)
distances, indices = nbrs.kneighbors(xyz)
r = np.mean(distances) + 0.1
# Precompute neighbors once
nbrs = NearestNeighbors(radius=r).fit(xyz)
all_neighbors = nbrs.radius_neighbors(xyz, return_distance=False)

eigenvalues_list = []
entropy_list = []
r = 1
start_time = time.time()

for i, point in enumerate(xyz):
    neighbors = all_neighbors[i]  # Get precomputed neighbors for the i-th point
    
    if len(neighbors) < 3:  # Ensure enough neighbors for covariance
        eigenvalues_list.append([0, 0, 0])
        entropy_list.append(np.inf)
        continue

    # Compute covariance matrix for the neighborhood
    neighborhood_points = xyz[neighbors]
    centered_points = neighborhood_points - np.mean(neighborhood_points, axis=0)
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Perform PCA to get eigenvalues
    pca = PCA(n_components=3)
    pca.fit(cov_matrix)
    eigenvalues = pca.explained_variance_
    print(i)

    # Compute entropy for this neighborhood
    entropy = compute_shannon_entropy(eigenvalues)

    # Append results
    eigenvalues_list.append(eigenvalues)
    entropy_list.append(entropy)


df_merged['Entropy'] = entropy_list
df_merged['Eigenvalues'] = eigenvalues_list
def classify_dimensionality(eigenvalues):
    l1, l2, l3 = eigenvalues
    if l1 > 2 * l2 and l1 > 2 * l3:  # Linear
        return '1D'
    elif l1 > 2 * l2 and l2 > 2 * l3:  # Planar
        return '2D'
    else:  # Volumetric
        return '3D'
    
# Classifying each point's dimensionality based on eigenvalues_list
dimensionality = [classify_dimensionality(ev) for ev in eigenvalues_list]

df_merged['Entropy'] = entropy_list
df_merged['Eigenvalues'] = eigenvalues_list

# Step 6: Add the dimensionality classification to the DataFrame
df_merged['Dimensionality'] = dimensionality

# Step 5: Sort points by entropy (for clustering later)
data_sorted = df_merged.sort_values(by='Entropy')

df_merged['Cluster'] = -1  # Initialize all points with no cluster (-1)
cluster_id = 0

# Step 3: Build clusters starting from the lowest-entropy points
for idx, row in data_sorted.iterrows():
    if df_merged.at[idx, 'Cluster'] == -1:  # If not already assigned to a cluster
        radius = r
        neighbors = get_neighbors_within_radius(idx, radius)
        
        # Only assign points with the same dimensionality to the cluster
        cluster_points = neighbors[df_merged.iloc[neighbors]['Dimensionality'] == row['Dimensionality']]
        
        # Assign these points to the current cluster
        df_merged.loc[cluster_points, 'Cluster'] = cluster_id
        
        # Move to the next cluster
        cluster_id += 1

# Step 4: Macro-Cluster Analysis - Remove small clusters
cluster_sizes = df_merged['Cluster'].value_counts()

# Define a threshold for small clusters (e.g., less than 10 points)
small_cluster_threshold = 10 ## TUNABLE PARAMETER!
small_clusters = cluster_sizes[cluster_sizes < small_cluster_threshold].index

# Mark points in small clusters as outliers
df_merged.loc[df_merged['Cluster'].isin(small_clusters), 'Predicted_Outlier'] = True


# Step 5: Micro-Cluster Analysis - Refine outliers based on eigenfeatures

def check_planar_cluster(cluster_points):
    # Check if points in a planar cluster match planarity expectations
    # Select only X, Y, Z coordinates for the covariance matrix
    planar_points = df_merged.loc[cluster_points, ['X', 'Y', 'Z']].values
    
    planarity_threshold = 0.5  # Threshold for acceptable planarity
    # Recompute planarity or check against precomputed planarity
    planarity = compute_planarity_for_cluster(planar_points)
    
    if planarity < planarity_threshold:
        print("Outlier in planar cluster due to low planarity")
    return planarity > planarity_threshold

def check_linear_cluster(cluster_points):
    # Check if points in a linear cluster match linearity expectations
    # Select only X, Y, Z coordinates for the covariance matrix
    linear_points = df_merged.loc[cluster_points, ['X', 'Y', 'Z']].values
    
    anisotropy_threshold = 0.7  # Threshold for acceptable anisotropy
    # Recompute anisotropy or check against precomputed anisotropy
    anisotropy = compute_anisotropy_for_cluster(linear_points)
    
    if anisotropy < anisotropy_threshold:
        print("Outlier in linear cluster due to low anisotropy")
    return anisotropy > anisotropy_threshold


# Iterate over clusters and refine based on micro-analysis
for cluster_id in df_merged['Cluster'].unique():
    cluster_points = df_merged.index[df_merged['Cluster'] == cluster_id].tolist()
    dimensionality = df_merged.loc[cluster_points[0], 'Dimensionality']
    
    if dimensionality == '2D':  # Planar surface
        if not check_planar_cluster(cluster_points):
            # Mark as outliers if they don't fit the planarity expectations
            df_merged.loc[cluster_points, 'Predicted_Outlier'] = True
    elif dimensionality == '1D':  # Linear surface
        if not check_linear_cluster(cluster_points):
            # Mark as outliers if they don't fit the anisotropy expectations
            df_merged.loc[cluster_points, 'Predicted_Outlier'] = True


y_true = np.where(df_merged['Tag'].isin(['Noise','Outlier']), 1, 0)  # 1 if Noise, else 0
y_pred = np.where(df_merged['Predicted_Outlier'].isin([True]), 1, 0)

tpr,fpr = indices_confusion_matrix(y_true, y_pred, 'Covariance method')

misclassified_outliers = df_merged[(y_pred == 1) & (y_true == 0)]
misclassified_tags = printPercentageMisclassifedV(misclassified_outliers)        
print("Punti rimossi per sbaglio:")
print(misclassified_tags)

misclassified_outliers = df_merged[(y_pred == 0) & (y_true == 1)]
misclassified_tags = printPercentageMisclassifedF(misclassified_outliers)
print("Punti che serviva rimuovere ma rip:")
print(misclassified_tags)

end_time = time.time()

# Compute the time taken
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")




















