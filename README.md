# Point-Clouds-Data-Cleaning
Outliers, Noise and Occlusions Removal in 6D Point Clouds for Improving Illegal Landfills Detection

## Abstract
Our master's thesis focuses on improving the analysis of illegal landfills by processing 6D point clouds obtained from passive (cameras) and active (Lidar) sensors. The key phases of our project include outliers detection, denoising, and occlusion removal (vegetation), which hinder waste detection and volume estimation. Our thesis contributes to the European project PERIVALLON, aiming to identify illegal landfills caused by environmental crimes related to pollution and waste dumping.

During our research, we analyzed various methods. For outlier removal and denoising, we employed techniques such as Statistical Outlier Removal (SOR), DBSCAN, Radius Outlier Removal (ROR), Bilateral Filtering, and the 3D Mean Shift Anisotropic filter. SOR proved effective in removing outliers, detecting over 90\% of anomalous points with a minimal number of misclassified points. Bilateral Filtering excelled in denoising, identifying approximately 59\% of noise points without mistakenly classifying waste pile points.

Vegetation removal, treated as a semantic segmentation problem, was addressed using the deep learning models RandLA-Net and PointNeXt, both based on PointNet++. PointNeXt outperformed RandLA-Net, achieving a Mean IoU score of 64\% and an Overall Accuracy above 95\% on a synthetic dataset created in Blender, which included labeled waste, vegetation, noise, and outliers across six scenarios and four types of waste: asbestos, metals, plastic waste, and tires. PointNeXt demonstrated outstanding performance in vegetation recognition, with an IoU score exceeding 96\%.
