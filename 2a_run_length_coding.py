"""
1D Run-Length Coding for Bit-Plane Images
Performs run-length encoding on bit plane images using 6-bit run values.
Assumes all rows start with 1 (if row starts with 0, first run length is 0).
"""

import os
import numpy as np
from PIL import Image


def run_length_encode_row(row):
    """
    Performs 1D run-length encoding on a binary row.
    Assumes row starts with 1. If it starts with 0, first run length is 0.
    Each run is represented with a 6-bit value (max run length = 63).
    
    Args:
        row: numpy array of binary values (0 or 255, or 0 or 1)
    
    Returns:
        List of run lengths (each 0-63)
    """
    # Convert to binary (0 or 1)
    binary_row = (row > 0).astype(np.uint8)
    
    runs = []
    
    # If row starts with 0, first run length is 0 (we assume starting with 1)
    if len(binary_row) == 0:
        return runs
    
    if binary_row[0] == 0:
        runs.append(0)  # Zero run of 1s to start
    
    # Now encode runs
    current_val = binary_row[0]
    current_run = 0
    
    for pixel in binary_row:
        if pixel == current_val:
            current_run += 1
            # If run exceeds 63 (6-bit max), split it
            if current_run > 63:
                runs.append(63)
                runs.append(0)  # Zero run of opposite value
                current_run = 1
        else:
            runs.append(min(current_run, 63))
            current_val = pixel
            current_run = 1
    
    # Append last run
    if current_run > 0:
        runs.append(min(current_run, 63))
    
    return runs


def run_length_encode_image(image_array):
    """
    Performs 1D run-length encoding on entire image (row by row).
    
    Args:
        image_array: 2D numpy array of binary image
    
    Returns:
        List of all run lengths for the image
    """
    all_runs = []
    for row in image_array:
        row_runs = run_length_encode_row(row)
        all_runs.extend(row_runs)
    return all_runs


def save_compressed(runs, filepath):
    """
    Saves run-length encoded data to binary file.
    Each run is stored as 6 bits, packed into bytes.
    """
    # Pack 6-bit values into bytes
    # 4 runs (24 bits) = 3 bytes
    packed_bytes = bytearray()
    
    bit_buffer = 0
    bits_in_buffer = 0
    
    for run in runs:
        bit_buffer = (bit_buffer << 6) | (run & 0x3F)
        bits_in_buffer += 6
        
        while bits_in_buffer >= 8:
            bits_in_buffer -= 8
            byte_val = (bit_buffer >> bits_in_buffer) & 0xFF
            packed_bytes.append(byte_val)
    
    # Flush remaining bits (pad with zeros)
    if bits_in_buffer > 0:
        byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
        packed_bytes.append(byte_val)
    
    with open(filepath, 'wb') as f:
        f.write(packed_bytes)
    
    return len(packed_bytes)


def process_folder(input_folder, output_folder, results_file):
    """
    Process all bit plane images in a folder.
    
    Args:
        input_folder: Path to folder containing bit plane images
        output_folder: Path to save compressed files
        results_file: Path to save compression ratios
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # Find all bit plane images
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png') and 'bit_plane' in f])
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        
        # Load image
        img = Image.open(img_path).convert('L')
        pixels = np.array(img)
        height, width = pixels.shape
        
        # Calculate uncompressed size (1 bit per pixel)
        uncompressed_bits = height * width
        
        # Perform run-length encoding
        runs = run_length_encode_image(pixels)
        
        # Calculate compressed size (6 bits per run)
        compressed_bits = len(runs) * 6
        
        # Save compressed file
        output_name = img_file.replace('.png', '.rl')
        output_path = os.path.join(output_folder, output_name)
        save_compressed(runs, output_path)
        
        # Calculate compression ratio
        compression_ratio = compressed_bits / uncompressed_bits
        
        results.append({
            'file': img_file,
            'uncompressed_bits': uncompressed_bits,
            'compressed_bits': compressed_bits,
            'num_runs': len(runs),
            'compression_ratio': compression_ratio
        })
        
        print(f"  {img_file}: ratio = {compression_ratio:.4f} ({len(runs)} runs)")
    
    # Save results to text file
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("          RUN-LENGTH ENCODING COMPRESSION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Encoding: 6-bit run values, 1D row-by-row, rows assumed to start with 1\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'File':<30} {'Uncomp Bits':<15} {'Comp Bits':<15} {'Ratio':<10}\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            f.write(f"{r['file']:<30} {r['uncompressed_bits']:<15} {r['compressed_bits']:<15} {r['compression_ratio']:.4f}\n")
        
        f.write("-" * 70 + "\n")
        
        # Calculate average
        avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
        f.write(f"\n{'Average Compression Ratio:':<45} {avg_ratio:.4f}\n")
        f.write("\nNote: Ratio < 1 means compression achieved, Ratio > 1 means expansion.\n")
    
    print(f"  Results saved to: {results_file}")
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process binary bit plane images
    print("\n" + "=" * 60)
    print("Processing Binary Bit Plane Images")
    print("=" * 60)
    
    binary_input = os.path.join(script_dir, 'bit_plane_images')
    binary_output = os.path.join(script_dir, 'RL_bit_plane_images')
    binary_results = os.path.join(binary_output, 'compression_ratios.txt')
    
    if os.path.exists(binary_input):
        binary_res = process_folder(binary_input, binary_output, binary_results)
    else:
        print(f"Error: {binary_input} not found!")
    
    # Process gray coded bit plane images
    print("\n" + "=" * 60)
    print("Processing Gray Coded Bit Plane Images")
    print("=" * 60)
    
    gray_input = os.path.join(script_dir, 'gray_coded_bit_plane_images')
    gray_output = os.path.join(script_dir, 'RL_gray_coded_bit_plane_images')
    gray_results = os.path.join(gray_output, 'compression_ratios.txt')
    
    if os.path.exists(gray_input):
        gray_res = process_folder(gray_input, gray_output, gray_results)
    else:
        print(f"Error: {gray_input} not found!")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    if 'binary_res' in dir() and 'gray_res' in dir():
        print(f"\n{'Bit Plane':<15} {'Binary Ratio':<15} {'Gray Ratio':<15} {'Difference':<15}")
        print("-" * 60)
        
        for b, g in zip(binary_res, gray_res):
            bit_num = b['file'].split('_')[-1].replace('.png', '')
            diff = b['compression_ratio'] - g['compression_ratio']
            better = "Gray better" if diff > 0 else "Binary better" if diff < 0 else "Equal"
            print(f"Bit {bit_num:<10} {b['compression_ratio']:<15.4f} {g['compression_ratio']:<15.4f} {diff:+.4f} ({better})")
        
        avg_binary = sum(r['compression_ratio'] for r in binary_res) / len(binary_res)
        avg_gray = sum(r['compression_ratio'] for r in gray_res) / len(gray_res)
        
        print("-" * 60)
        print(f"{'Average':<15} {avg_binary:<15.4f} {avg_gray:<15.4f} {avg_binary - avg_gray:+.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
