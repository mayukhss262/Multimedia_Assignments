"""
Bit-Plane Extraction Script
Extracts all 8 bit planes from lena.png and saves them as images.
"""

import os
import numpy as np
from PIL import Image


def get_bit_plane(image_array, bit_position):
    """
    Extracts a specific bit plane from the image.
    
    Args:
        image_array: numpy array of pixel values
        bit_position: which bit to extract (0 = LSB, 7 = MSB)
    
    Returns:
        Binary image (0 or 255) showing the bit plane
    """
    mask = 1 << bit_position
    bit_plane = ((image_array & mask) >> bit_position) * 255
    return bit_plane.astype(np.uint8)


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(script_dir, 'lena.png')
    output_dir = os.path.join(script_dir, 'bit_plane_images')
    
    # Check if input exists
    if not os.path.exists(input_image):
        print(f"Error: {input_image} not found.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Load image and convert to grayscale
    print(f"Loading {input_image}...")
    img = Image.open(input_image).convert('L')  # 'L' = grayscale
    pixels = np.array(img)
    print(f"Image size: {img.width}x{img.height}")
    
    # Extract and save bit planes
    print("\n--- Extracting Bit Planes ---")
    for i in range(8):
        bit_plane = get_bit_plane(pixels, i)
        output_path = os.path.join(output_dir, f'bit_plane_{i}.png')
        Image.fromarray(bit_plane).save(output_path)
        print(f"  Saved: bit_plane_{i}.png (bit {i})")
    
    print(f"\nDone! All images saved to: {output_dir}")


if __name__ == "__main__":
    main()
