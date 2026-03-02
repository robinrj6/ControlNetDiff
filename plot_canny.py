"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

Source: https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html
"""

import os
import sys
import argparse
import numpy as np
from skimage import feature, io, filters
from pathlib import Path


def estimate_sigma(image):
    """
    Estimate optimal sigma based on image noise level.
    Uses Laplacian variance to detect noise and adjust sigma accordingly.
    
    Args:
        image: Input image (grayscale, normalized to 0-1)
        
    Returns:
        float: Estimated sigma value
    """
    # Apply Laplacian filter to detect edges/noise
    laplacian = filters.laplace(image)
    laplacian_var = laplacian.var()
    
    # Map variance to sigma (lower variance = less noise = lower sigma needed)
    # Empirically tuned for typical images
    if laplacian_var < 0.001:
        sigma = 0.5  # Very clean image
    elif laplacian_var < 0.01:
        sigma = 1.0  # Clean image
    elif laplacian_var < 0.1:
        sigma = 1.5  # Moderate noise
    elif laplacian_var < 0.5:
        sigma = 2.5  # Noisy image
    else:
        sigma = 3.5  # Very noisy image
    
    return sigma


def process_images(input_folder, output_folder, sigma=None, auto_sigma=False):
    """
    Process all images in input_folder and save their Canny edges to output_folder.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where edge images will be saved
        sigma: Standard deviation for Gaussian kernel (default: None)
        auto_sigma: If True, automatically detect sigma for each image (default: False)
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image(s) to process...")
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_canny.png"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Read image (convert to grayscale if needed)
            image = io.imread(input_path)
            if len(image.shape) == 3:
                # Convert RGB to grayscale
                image = np.mean(image, axis=2)
            
            # Normalize to 0-1 range
            image = image.astype(float) / 255.0 if image.max() > 1 else image.astype(float)
            
            # Determine sigma
            if auto_sigma:
                current_sigma = estimate_sigma(image)
                print(f"  → Estimated sigma: {current_sigma:.2f}")
            else:
                current_sigma = sigma if sigma is not None else 1.0
            
            # Apply Canny edge detection
            edges = feature.canny(image, sigma=current_sigma)
            
            # Convert to 0-255 range and save
            edges_uint8 = (edges * 255).astype(np.uint8)
            io.imsave(output_path, edges_uint8)
            
            print(f"✓ Processed: {filename} -> {output_filename}")
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Canny edge detection to images in a folder")
    parser.add_argument("input_folder", help="Path to input folder containing images")
    parser.add_argument("output_folder", help="Path to output folder for edge images")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma for Canny filter (default: 1.0)")
    parser.add_argument("--auto-sigma", action="store_true", help="Automatically detect sigma based on image noise")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    process_images(args.input_folder, args.output_folder, args.sigma, args.auto_sigma)
