import os
import numpy as np
from PIL import Image


def binary_to_gray(image_array):
    
    return image_array ^ (image_array >> 1)


def get_bit_plane(image_array, bit_position):
    
    mask = 1 << bit_position
    bit_plane = ((image_array & mask) >> bit_position) * 255
    return bit_plane.astype(np.uint8)


def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(script_dir, 'lena.png')
    output_dir = os.path.join(script_dir, 'gray_coded_bit_plane_images')
    
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
    
    # Convert to Gray code
    gray_coded_pixels = binary_to_gray(pixels)
    
    gray_code_image_path = os.path.join(output_dir, 'gray_code_image.png')
    Image.fromarray(gray_coded_pixels.astype(np.uint8)).save(gray_code_image_path)
    
    # Extract and save bit planes 
    print("\n--- Extracting Gray Coded Bit Planes ---")
    for i in range(8):
        bit_plane = get_bit_plane(gray_coded_pixels, i)
        output_path = os.path.join(output_dir, f'gray_bit_plane_{i}.png')
        Image.fromarray(bit_plane).save(output_path)
        print(f"  Saved: gray_bit_plane_{i}.png (bit {i})")
    
    print(f"\nAll images saved to: {output_dir}")


if __name__ == "__main__":
    main()
