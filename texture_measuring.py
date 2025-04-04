import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_leaf_texture(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found! ({image_path})")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute texture descriptors
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # Compute Laplacian variance for smoothness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # Compute simple homogeneity measure
    homogeneity = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
    
    # Determine texture status based on smoothness
    smoothness_threshold = 190  # Adjust based on dataset
    homogeneity_threshold = 5  # Adjust based on dataset
    
    texture_status = "Smooth Texture" if laplacian_var < smoothness_threshold and homogeneity < homogeneity_threshold else "Rough Texture"
    
    print(f"\nAnalyzing: {image_path}")
    print(f"Mean Intensity: {mean_intensity:.2f}")
    print(f"Standard Deviation of Intensity: {std_intensity:.2f}")
    print(f"Laplacian Variance: {laplacian_var:.2f}")
    print(f"Homogeneity (Sobel Mean): {homogeneity:.2f}")
    print(f"Texture Status: {texture_status}")

for i in range(1, 11): 
    image_path = f'img/Day-{i}/07.00am/B/NF2-{i}.1.jpg'
    analyze_leaf_texture(image_path)
    image_path = f'img/Day-{i}/07.00pm/B/NF2-{i}.3.jpg'  
    analyze_leaf_texture(image_path)
   