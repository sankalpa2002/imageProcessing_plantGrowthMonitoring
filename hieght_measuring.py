import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_plant_height(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open {image_path}")
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green color range for masking
    lower_green = np.array([15, 60, 0])  
    upper_green = np.array([90, 255, 255])

    # Create a mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    mask = cv2.medianBlur(mask, 5)  # Smooth the mask

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set minimum contour area to filter small detections
    min_contour_area = 500

    max_height = 0
    min_y, max_y = np.inf, 0

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue  # Skip small contours

        _, y, _, h = cv2.boundingRect(contour)
        if h > max_height:
            max_height = h
            min_y = min(min_y, y)
            max_y = max(max_y, y + h)

    if max_height > 0:
        height_value = max_y - min_y
        return height_value  # Return height in pixels
    else:
        return None  # No plant detected

# **Step 2: Store the Heights for Graph**
days = []
heights = []

# Loop through Day (1).jpg to Day (10).jpg and store height
for i in range(1, 11):
    image_path_am = f'img/day-{i}/07.00am/A/NF1-{i}.1.jpg' 
    height_am = calculate_plant_height(image_path_am)

    image_path_pm = f'img/day-{i}/07.00pm/A/NF1-{i}.3.jpg' 
    height_pm = calculate_plant_height(image_path_pm)

    # Take the average height of AM & PM if both exist
    if height_am is not None and height_pm is not None:
        height = (height_am + height_pm) / 2
    elif height_am is not None:
        height = height_am
    elif height_pm is not None:
        height = height_pm
    else:
        height = None  # No data available for this day

    if height is not None:
        days.append(i)
        heights.append(height)
        print(f"Day {i}: Average Height = {height} pixels")

#Plot the Graph
plt.figure(figsize=(8, 5))
plt.plot(days, heights, marker='o', linestyle='-', color='b', label='Plant Height')

plt.xlabel("Day")
plt.ylabel("Height (Pixels)")
plt.title("Plant Growth Over Time")
plt.legend()
plt.grid(True)

# Save or display the graph
print(  )
plt.savefig("plant_growth_graph.png")  # Saves the graph as an image
plt.show()
