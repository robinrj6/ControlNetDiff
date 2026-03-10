import numpy as np
import cv2 as cv
import os

def canny_edge_detection(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    # Higher thresholds to detect only strong edges (outlines)
    edges = cv.Canny(img, 200, 400, apertureSize=3)
    return edges

def main():
    image_path = './test/'  # Replace with your image path
    output_path = './edges/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # get all the images in the directory
    image_count = 0
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg') :
            edges = canny_edge_detection(os.path.join(image_path, filename))
            if edges is not None:
                cv.imwrite(os.path.join(output_path, filename), edges)
                image_count += 1
                print(f"Processed: {filename}")
    
    print(f"Total images processed: {image_count}")

if __name__ == "__main__":
    main()
